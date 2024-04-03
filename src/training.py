import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

def build_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def compile_model(model):
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

def train_model(model, train_data, val_data, log_dir, epochs=20):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    hist = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[tensorboard_callback])
    return hist
