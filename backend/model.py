###final
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import os

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)

# === 1. Load and preprocess images ===
def preprocess_custom_image(img_path):
    img = Image.open(img_path).convert('L')  # Grayscale
    img = img.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img)
    if np.mean(img_array) > 127:  # Invert white background
        img = ImageOps.invert(img)

    img = ImageEnhance.Contrast(img).enhance(2.0)

    # Add padding for dot visibility (e.g., in 'i')
    padded_img = Image.new('L', (32, 32), color=0)
    padded_img.paste(img, (2, 2))
    padded_img = padded_img.resize((28, 28), Image.LANCZOS)

    return np.array(padded_img).astype('float32') / 255.0

custom_dir = "/home/fleettrack/subhaashree/alphabet_recognition/custom_dataset"
x_data, y_data = [], []

classes = sorted([d for d in os.listdir(custom_dir) if os.path.isdir(os.path.join(custom_dir, d))])
class_to_idx = {label: idx for idx, label in enumerate(classes)}

for label in classes:
    label_dir = os.path.join(custom_dir, label)
    for file in os.listdir(label_dir):
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            try:
                img_path = os.path.join(label_dir, file)
                img_array = preprocess_custom_image(img_path)
                x_data.append(img_array)
                y_data.append(class_to_idx[label])
            except Exception as e:
                print(f"Error loading {file}: {e}")

x_data = np.array(x_data).reshape(-1, 28, 28, 1)
y_data = to_categorical(y_data, num_classes=len(classes))
print("Original dataset:", x_data.shape)

# === 2. Balance dataset (1000 per class) ===
def balance_dataset(x, y, samples_per_class=1000):
    y_labels = np.argmax(y, axis=1)
    x_bal, y_bal = [], []
    for c in range(len(classes)):
        idx = np.where(y_labels == c)[0]
        count = len(idx)
        if count == 0:
            print(f"❗ No samples for class '{classes[c]}'")
            continue
        if count < samples_per_class:
            idx = np.repeat(idx, (samples_per_class // count) + 1)[:samples_per_class]
        else:
            idx = idx[:samples_per_class]
        x_bal.append(x[idx])
        y_bal.append(y[idx])
    return np.concatenate(x_bal), np.concatenate(y_bal)

x_data, y_data = balance_dataset(x_data, y_data, samples_per_class=1000)
print("Balanced dataset:", x_data.shape)

# === 3. Split ===
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.1, stratify=np.argmax(y_data, axis=1), random_state=42
)

# === 4. Augmentation ===
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)
datagen.fit(x_train)

# === 5. Model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (2, 2), activation='relu'),
    BatchNormalization(),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === 6. Callbacks ===
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_alphabet_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# === 7. Class weights (optional for better 'i' vs 'l') ===   
y_train_labels = np.argmax(y_train, axis=1) # convert one-hot encode[0 0 0 1 0...] to normal labels[2,4,3,6,...]
weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels) #compute weight for all classes from 1 to 25
class_weights = dict(enumerate(weights))#use dictionary to map weights of each class

# === 8. Train ===
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, lr_schedule, checkpoint],
    class_weight=class_weights  # Include this line for better class balance
)

# === 9. Per-class Accuracy ===
def evaluate_per_class(x, y):
    preds = model.predict(x)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y, axis=1)
    for i, label in enumerate(classes):
        mask = true_labels == i
        acc = np.mean(pred_labels[mask] == i) if mask.sum() > 0 else 0
        print(f"Class '{label}': {acc:.4f}")

evaluate_per_class(x_val, y_val)

# === 10. Plot History ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("alphabet_training_history.png")
plt.close()

# === 11. Save Final Model ===
model.save("alphabets_model1.keras")
print("\n✅ Final model saved as 'alphabets_model1.keras'")
