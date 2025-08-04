from flask import Flask, request, jsonify
from datetime import datetime
from PIL import Image
import numpy as np
import os
import uuid
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the updated trained model
model = load_model("alphabets_model_final.keras")

# EMNIST class labels: 0 = 'a', ..., 25 = 'z'
class_names = [chr(i) for i in range(97, 123)]  # ['a', ..., 'z']

# Folder to store corrected images
DATASET_DIR = "/home/fleettrack/subhaashree/alphabet_recognition/custom_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
for label in class_names:
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)


# ------------------- Updated Preprocessing Function ------------------- #
def preprocess_character(pil_img):
    gray = pil_img.convert("L")
    img_np = np.array(gray)

    # Invert to make background black if it's white
    if np.mean(img_np) > 127:
        img_np = 255 - img_np

    # Threshold to binary
    _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get bounding box
    coords = cv2.findNonZero(thresh)
    if coords is None:
        raise ValueError("No character found in image.")
    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]

    # Add 25% padding to preserve small components (e.g., dot on 'i')
    padding = max(w, h) // 4
    padded = cv2.copyMakeBorder(
        cropped,
        top=padding,
        bottom=padding,
        left=padding,
        right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    # Resize to 28x28
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize
    normalized = resized.astype("float32") / 255.0
    normalized = np.expand_dims(normalized, axis=-1)  # Shape: (28, 28, 1)

    return normalized


# ------------------- /predict Endpoint ------------------- #
@app.route("/predict", methods=["POST"])
def predict_character():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")

        processed = preprocess_character(image)
        processed = np.expand_dims(processed, axis=0)  # (1, 28, 28, 1)

        prediction = model.predict(processed)[0]
        index = int(np.argmax(prediction))
        confidence = float(prediction[index])
        predicted_char = class_names[index]

        return jsonify({
            "type": "single",
            "prediction": predicted_char,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------- /correct Endpoint (No Change) ------------------- #
@app.route("/correct", methods=["POST"])
def correct():
    if 'image' not in request.files or 'label' not in request.form:
        return jsonify({"error": "Missing image or label"}), 400

    label = request.form['label'].lower()
    if not (label.isalpha() and len(label) == 1 and label in class_names):
        return jsonify({"error": "Invalid label"}), 400

    try:
        file = request.files['image']
        pil_img = Image.open(file).convert("L")  # grayscale

        # Convert to numpy
        img_np = np.array(pil_img)

        # Invert if background is white
        if np.mean(img_np) > 127:
            img_np = 255 - img_np

        # Threshold
        _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Bounding box
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return jsonify({"error": "No character found"}), 400
        x, y, w, h = cv2.boundingRect(coords)
        cropped = thresh[y:y+h, x:x+w]

        # Padding
        padding_ratio = 0.2
        pad = int(max(w, h) * padding_ratio)
        padded = cv2.copyMakeBorder(
            cropped,
            pad, pad, pad, pad,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )

        # Center on square canvas
        h2, w2 = padded.shape
        max_side = max(h2, w2)
        square = np.zeros((max_side, max_side), dtype=np.uint8)
        y_offset = (max_side - h2) // 2
        x_offset = (max_side - w2) // 2
        square[y_offset:y_offset + h2, x_offset:x_offset + w2] = padded

        # Resize to 28x28 without blur
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_NEAREST)

        # Save
        save_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex + ".png"
        save_path = os.path.join(save_dir, filename)
        Image.fromarray(resized).save(save_path)

        return jsonify({
            "message": "Image saved successfully",
            "label": label,
            "path": save_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ------------------- Run the App ------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9010)
