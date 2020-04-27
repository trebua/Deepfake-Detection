'''
An API for predicting an inputted video from path or youtube url. Outputs prediction and confidence.
'''
from keras.models import model_from_json
import pathlib
import cv2
import cvlib as cv
import random
import numpy as np
import os
import pafy

# Load model
current_dir = f'{str(pathlib.Path(__file__).parent.absolute())}'
with open(f'{current_dir}/model/model.json', 'r') as json_file:
    model = model_from_json(json_file.read())
    model.load_weights(f'{current_dir}/model/model.h5')

def get_votes(video, model=model, sample=19):
    '''
    Samples a video capture into frames, run prediction with model on frames and returns a list of the predictions (votes)

    Args:
        video: a opencv video capture
        model: a keras model for binary classification
        sample: how many frames should be sampled for face extraction

    Returns:
        votes: the predictions for each frame. ex. [1,0,1,1,0,..]
    '''
    votes = []
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = random.sample([i for i in range(frame_count)], min(sample, frame_count))
    for i in frames:
        detections = []
        video.set(1,i)
        _, frame = video.read() 
        faces, _ = cv.detect_face(frame, threshold=0.9)
        for j, (x0,y0,x1,y1) in enumerate(faces):
            try:
                face = frame[y0:y1, x0:x1]
                face = cv2.resize(face, (100,100), interpolation = cv2.INTER_AREA)
                face = face.astype(np.float32)
                face /= 255
                detections.append(face)
            except Exception:
                pass
        vote = 0
        for face in detections:
            face = face.reshape((1,len(face[0]),len(face[1]),3))
            prediction = model.predict(face)[0][0]
            vote += int(prediction >= 0.5)
        votes.append(int(vote >= 1))
    return votes

def get_prediction(votes, id, is_youtube):
    '''
    Aggregates the votes (predictions) and outputs an answer

    Args:
        votes: predictions
        id: path or url
        is_youtube: if it is a youtube video
    
    Returns:
        (REAL/FAKE, confidence%)
    '''
    reals = votes.count(1)
    fakes = votes.count(0)
    if reals >= fakes:
        result = 'REAL'
        conf = (reals/(reals+fakes))*100
    else:
        result = 'FAKE'
        conf = (fakes/(reals+fakes))*100
    if is_youtube:
        print(f'{id}: {result} {round(conf,2)}%')
    else: 
        video_name = os.path.basename(id)
        print(f'{video_name}: {result} {round(conf,2)}%')
    return result, conf


def predict_video(path):
    '''
    Classifies video in path as real or fake.

    Args:
        path: path to the video
    
    Returns:
        (REAL/FAKE, confidence%)
    '''
    video = cv2.VideoCapture(path)
    votes = get_votes(video)
    return get_prediction(votes,path,is_youtube=False)

def predict_youtube(url):
    '''
    Classifies youtube video with url as real or fake

    Args:
        url: url of the youtube video

    Returns:
        (REAL/FAKE, confidence%)
    '''

    vPafy = pafy.new(url)
    name = f'{vPafy.title} ({url})'
    play = vPafy.getbestvideo()
    video = cv2.VideoCapture(play.url)
    votes = get_votes(video)
    return get_prediction(votes, name, is_youtube=True)
