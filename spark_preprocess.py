'''
Distributed preprocessing; can be run locally, standalone or in a cluster.

Important to set correct memory configurations, a working example:
spark-submit 
--conf spark.driver.memory=8g --conf spark.executor.memory=8g --conf spark.memory.offHeap.enabled=true 
--conf spark.memory.offHeap.size=8g 
--conf spark.driver.maxResultSize=8g
--master local[*] spark_preprocess.py
'''

import pathlib
import pyspark
import os
import cv2
import random
import cvlib as cv
import numpy as np

sc = pyspark.SparkContext('local[*]')

# Paths needed for reading and storing data
current_dir = str(pathlib.Path(__file__).parent.absolute())
real_path = f"{current_dir}/data/original_sequences/actors/c23/videos/"
fake_path = f"{current_dir}/data/manipulated_sequences/DeepFakeDetection/c23/videos/"

def get_frames(label_path, sample=5, face_threshold=0.9, dimensions=(100,100), ratio=9):
    '''
    Function used in the map phase for each video.

    Args:
        label_path: tuple consisting of (label 1 or 0, path to video file)
        sample: how many frames to extract from a video
        face_threshold: how strict the face extraction is, 1 is strictest.
        dimensions: the dimensions of the extracted face to be stored
        ratio: fakes/reals in order to balance the sampling

    Returns: 
        A list of (label, face) pairs
    '''
    result = []
    label, path = label_path
    if label == 1:  # Add more samples if the video is labeled with the minority class
        sample *= ratio
    video = cv2.VideoCapture(path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Shuffles frames in order to pick randomly
    frames = random.sample([i for i in range(frame_count)], min(sample, frame_count))   
    for frame_index in sorted(frames):
        video.set(1,frame_index)
        _, frame = video.read()
        try:
            faces, confidences = cv.detect_face(frame, threshold=face_threshold)
            for j, (x0,y0,x1,y1) in enumerate(faces):
                face = frame[y0:y1, x0:x1]
                if len(face) > 0 and len(face[0] > 0):
                    face = cv2.resize(face, dimensions, interpolation = cv2.INTER_AREA)
                    face = face.astype(np.float32)
                    face /= 255
                    result.append((label, face))
        except Exception:
            pass
    video.release() 
    cv2.destroyAllWindows()
    global processed; global workers; processed += 1; print(f'{processed}/{total//workers} processed.')
    return result

#Gets label-path tuples for reals and fakes
reals = [(1, real_path + '/' + path) for path in os.listdir(real_path)]
fakes = [(0, fake_path + '/' + path) for path in os.listdir(fake_path)]
videos = reals + fakes

#Data for keeping track of progress
total = len(videos)
processed = 0
workers = 8

#Paralellizes the videos, maps the (label, frame) to [(label, face1), (label, face2),..] and reduces to one list of label-face tuples
rdd = sc.parallelize(videos)
label_face = rdd.map(lambda label_path: get_frames(label_path)).reduce(lambda res1, res2: res1 + res2)
y, X = zip(*label_face)

#Saves the faces in X and labels in y
np.save(f'{current_dir}/X', np.array(X))
np.save(f'{current_dir}/y', np.array(y))