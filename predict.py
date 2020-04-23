from keras.models import model_from_json
import pathlib
import cv2
import cvlib as cv
import random
import numpy as np
import os
import argparse

# Load model
current_dir = f'{str(pathlib.Path(__file__).parent.absolute())}'
with open(f'{current_dir}/model/model.json', 'r') as json_file:
    model = model_from_json(json_file.read())
    model.load_weights(f'{current_dir}/model/model.h5')

def predict_video(path, model=model, sample=29):
    '''
    Takes in a video path and samples faces from it. 
    The model predicts whether the individual frames are real or fake, and the votes are aggregated to
    output a decision of whether the video is real or fake

    Args:
        path: path to the video
        model: which model to use for the prediction
        sample: how many frames will be attempted to be extracted
    '''
    votes = []
    video = cv2.VideoCapture(path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = random.sample([i for i in range(frame_count)], min(sample, frame_count))
    for i in frames:
        detections = []
        video.set(1,i)
        _, frame = video.read() 
        faces, _ = cv.detect_face(frame, threshold=0.9)
        for j, (x0,y0,x1,y1) in enumerate(faces):
            face = frame[y0:y1, x0:x1]
            face = cv2.resize(face, (100,100), interpolation = cv2.INTER_AREA)
            face = face.astype(np.float32)
            face /= 255
            detections.append(face)
        vote = 0
        for face in detections:
            face = face.reshape((1,len(face[0]),len(face[1]),3))
            prediction = model.predict(face)[0][0]
            vote += int(prediction >= 0.5)
        votes.append(int(vote >= 1))
    reals = votes.count(1)
    fakes = votes.count(0)
    if reals >= fakes:
        result = 'REAL'
        conf = (reals/(reals+fakes))*100
    else:
        result = 'FAKE'
        conf = (fakes/(reals+fakes))*100
    video_name = os.path.basename(path)
    print(f'{video_name}: {result} {round(conf,2)}%')


my_parser = argparse.ArgumentParser(description='Determine whether a video is real or fake.')
my_parser.add_argument('--path','-p', action='store', type=str, help='The path to the video to be determined.')
args = my_parser.parse_args()
path = args.path
predict_video(path)