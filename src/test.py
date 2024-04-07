# src/test.py

import tensorflow as tf
from src.preprocessing import preprocess_data
from src.model import build_model

# Load data
data_dir = '../data'
data = tf.keras.utils.image_dataset_from_directory(data_dir)

# Preprocess data
data = preprocess_data(data)

# Load model
model = tf.keras.models.load_model('../models/trained_model.h5')

# Evaluate model
metrics = model.evaluate(data)
print("Test Accuracy:", metrics[1])
