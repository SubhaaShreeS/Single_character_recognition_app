# character_recognition_with_update

This project is a full-stack application for recognizing handwritten lowercase alphabet characters (a–z). 
It consists of:
Flask Backend: A Python-based API that uses a pre-trained Keras model to predict characters from images and store corrected images in a dataset.

Flutter Frontend: A mobile app where users can draw characters, get predictions, and submit corrections.

Training Script: A Python script to train the Keras model using a custom dataset, with data balancing and augmentation.

The system allows users to draw a letter, predict it using a neural network, and correct wrong predictions to improve the model by adding images to the dataset.
