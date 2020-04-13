import numpy as np
import os
import cv2
import pathlib
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

'''
A TEMPORARY and simple CNN to test the preprocessing

Prerequisties:
Keras
Tensorflow
Sklearn
Numpy
CV2
'''

data_dir = f'{str(pathlib.Path(__file__).parent.absolute())}/processed/'
X = np.load(f'{data_dir}X.npy')
y = np.load(f'{data_dir}y.npy')
x_dim, y_dim = len(X[0]), len(X[0][0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier = Sequential()
classifier.add(Convolution2D(32, (3,3), input_shape=(x_dim,y_dim,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=10)
print(classifier.evaluate(X_test, y_test))

