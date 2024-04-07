# preprocess the data
import tensorflow as tf
import os
import cv2  
import imghdr 

def remove_invalid_images(data_dir, image_exts):
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

def preprocess_data(data):
    data = data.map(lambda x,y: (x/255, y))
    return data
