# train the already created model on a new dataset
import tensorflow as tf
from src.preprocessing import remove_invalid_images, preprocess_data
from src.model import load_model, compile_model

# Out Of Memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Directory of the new dataset
new_data_dir = '../data'
logdir = '../logs'

image_exts = ['jpg', 'jpeg']

remove_invalid_images(new_data_dir, image_exts)

# Load the already trained model
trained_model_path = 'models/trainedModel.h5'
model = load_model(trained_model_path)

# Compile the model
model = compile_model(model)

# Load data from the new dataset directory
new_data = tf.keras.utils.image_dataset_from_directory(new_data_dir)
new_data = preprocess_data(new_data)

# Train model on the new dataset
model.fit(new_data, epochs=20)

# Save the updated model
model.save('models/newtrainedModel.h5')
