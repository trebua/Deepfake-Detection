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

x, y = [],[]

data_dir = f'{str(pathlib.Path(__file__).parent.absolute())}/processed/'
for label in ['fake', 'real']:
    path = data_dir + label + '/'
    for img_name in os.listdir(path):
        img_path = path + img_name
        img = cv2.imread(img_path)
        x.append(np.array(img))
        y.append(int(label == 'real'))

x = np.array(x)
x = x.astype(np.float32)
x /= 255
y = np.array(y)
x_dim, y_dim = len(x[0]), len(x[0][0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

classifier = Sequential()
classifier.add(Convolution2D(32, (3,3), input_shape=(x_dim,y_dim,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, epochs=10)
print(classifier.evaluate(x_test, y_test))

