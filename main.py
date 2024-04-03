import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from src.preprocessing import remove_invalid_images, preprocess_data
from src.training import build_model, compile_model, train_model
from src.evaluation import evaluate_model

# Out Of Memory errors
gpus= tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data'
logdir = 'logs'

image_exts = ['jpg', 'jpeg']

remove_invalid_images(data_dir, image_exts)

data = image_dataset_from_directory(data_dir)
data = preprocess_data(data)

train_size = int(len(data))
val_size = int(len(data) * 0.2) + 1
test_size = int(len(data) * 0.1) + 1

train_data = data.take(train_size)
val_data = data.skip(train_size).take(val_size)
test_data = data.skip(train_size + val_size).take(test_size)

model = build_model()
model = compile_model(model)
hist = train_model(model, train_data, val_data, logdir)

precision, recall, accuracy = evaluate_model(model, test_data)
print(f'Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}')
