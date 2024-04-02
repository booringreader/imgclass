import tensorflow as tf
import os
import cv2  
import imghdr #for checking file extensions
import numpy as np
from matplotlib import pyplot as plt

# Out Of Memory errors
gpus= tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = '~/Developer/personal/halide/data'

image_exts = ['jpg', 'jpeg']

# removing dodgy images
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('image not found'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('invalid extension'.format(image_path))

# load data
tf.Data.Dataset
data = tf.keras.utils.image_dataset_from_directory('data')

data_iterator = data.as_numpy_iterator()