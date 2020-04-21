import pathlib
import pyspark
import os
import cv2
import random
import cvlib as cv
import numpy as np
from time import time

sc = pyspark.SparkContext('local[*]')

# Paths needed for reading and storing data
current_dir = str(pathlib.Path(__file__).parent.absolute())
real_path = f"{current_dir}/data/original_sequences/actors/c40/videos/"
fake_path = f"{current_dir}/data/manipulated_sequences/DeepFakeDetection/c40/videos/"

def get_frames(label_path, sample=1, face_threshold=0.9, dimensions=(10,10)):
    global total; global processed; global avg_time
    start_time = time()
    result = []
    label, path = label_path
    video = cv2.VideoCapture(path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
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
    time_spent = time()-start_time
    avg_time = (avg_time*(processed)+time_spent)/(processed+1)
    remaining = total-processed
    processed += 1; print(f'{processed}/{total} processed. {int(avg_time*remaining)}s left.', end='\r')
    return result

reals = [(1, real_path + '/' + path) for path in os.listdir(real_path)]
fakes = [(0, fake_path + '/' + path) for path in os.listdir(fake_path)]
movies = reals + fakes
total = len(movies)
processed, avg_time = 0, 0
rdd = sc.parallelize(movies,1)
label_face = rdd.map(lambda label_path: get_frames(label_path)).reduce(lambda res1, res2: res1 + res2)
X, y = zip(*label_face)
np.save(f'{current_dir}X1', np.array(X))
np.save(f'{current_dir}y1', np.array(y))