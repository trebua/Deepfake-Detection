'''
Face detection based on the tutorial https://realpython.com/face-recognition-with-python/.
'''

import cv2
import pathlib

current_path = str(pathlib.Path(__file__).parent.absolute())
cascade_path =  current_path + '/haarcascade_frontalface_default.xml'

def get_faces(img, min_size=(80,80), scale_factor=1.3):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=scale_factor,
        minNeighbors=5,
        minSize=min_size,
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    return faces