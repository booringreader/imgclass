import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from src.preprocessing import remove_invalid_images, preprocess_data
from src.model import build_model, compile_model, train_model # useful when building the model for the first time
from src.evaluation import evaluate_model

# Out Of Memory errors
gpus= tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data'
logdir = 'logs'
test_data_dir = 'testing' # dataset directory for testing the model

image_exts = ['jpg', 'jpeg']

remove_invalid_images(data_dir, image_exts)
remove_invalid_images(test_data_dir, image_exts)

# for training and validation
data = image_dataset_from_directory(data_dir)
data = preprocess_data(data)

# for testing
test_data = image_dataset_from_directory(test_data_dir)
test_data = preprocess_data(test_data)

train_size = int(len(data)*0.8)
val_size = int(len(data) * 0.2) + 1
test_size = int(len(test_data))

train_data = data.take(train_size)
val_data = data.skip(train_size).take(val_size)
test_data = data.skip(train_size + val_size).take(test_size)

model = build_model()
model = compile_model(model)
hist = train_model(model, train_data, val_data, logdir)

model.save('models/trainedModel.h5')
precision, recall, accuracy = evaluate_model(model, test_data)
print(f'Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}')
