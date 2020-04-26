'''
Creates the CNN and trains it on the preprocessed dataset - then it is stored in the model-folder.

This can also be done in the train_model.ipynb notebook - with plotting
'''

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D
from keras.optimizers import Adam
from sklearn import metrics
import cv2
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pathlib
import pydot

# Load labelled and preprocessed dataset
current_dir = str(pathlib.Path(__file__).parent.absolute())
x = np.load(f'{current_dir}/X.npy')
y = np.load(f'{current_dir}/y.npy')

# Random seeded shuffling
np.random.seed(2)
p = np.random.permutation(x.shape[0])
x = x[p]
y = y[p]

# Split into testing and training
x_test, y_test = x[:3100], y[:3100]
x_train, y_train = x[3100:], y[3100:]

# Create the CNN architecture
model = Sequential()
model.add(Conv2D(8, kernel_size=3, activation='relu', input_shape=(100, 100, 3)))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D())
model.add(Conv2D(8, kernel_size=5, activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D())
model.add(Conv2D(16, kernel_size=5, activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D())
model.add(Conv2D(16, kernel_size=5, activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile and train model
model.compile(optimizer=Adam(learning_rate=0.0000005, decay=1e-2), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=256, epochs=100)

# Evaluates accuracy and loss
print(model.evaluate(x_test, y_test))

#Save model
with open(f'{current_dir}/model/model.json', 'w') as json_file:
    json_file.write(model.to_json())
    model.save_weights(f'{current_dir}/model/model.h5')