import os
import cv2
import random
import pathlib
from time import time
import cvlib as cv

# Paths needed for reading and storing data
current_dir = str(pathlib.Path(__file__).parent.absolute())
data_dir = '/data'
processed_dir = '/processed/'
datasets = {
    'real': '/original_sequences/actors/c40/videos/',
    'fake': '/manipulated_sequences/DeepFakeDetection/c40/videos/'
}


def create_processed_dirs():
    '''
    Creates the directory structures processed/fake and processed/real

    Returns:
        path to the folder where to processed frames will be stored
    '''
    processed_path = current_dir + processed_dir
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    real_dir = processed_path + 'real'
    fake_dir = processed_path + 'fake'
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
    if not os.path.exists(fake_dir):
        os.makedirs(fake_dir)
    return processed_path


def preprocess_videos(dimensions=(150,150), sample=2, count=False):
    '''
    Runs through all the videos downloaded and saves 'samples' amount of frames with 'dimensions' dimensions.

    Returns
        None
    '''
    processed_path = create_processed_dirs()
    for label, label_path in datasets.items():  # Loop through labels (real, fake) with their corresponding dataset paths
        processed = 0
        path = current_dir + data_dir + label_path
        videos = os.listdir(path)
        avg_time = 0
        for c, video_name in enumerate(videos):
            start_time = time()
            video_path = path + video_name
            video = cv2.VideoCapture(video_path)
            image_path = processed_path + label + '/' + video_name.split('.')[0]

            # Saves faces from a random sampled subset of the video
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            sampled_frames = random.sample([i for i in range(frame_count)], sample) 
            for i in sampled_frames:
                video.set(1,i)
                name = f'{image_path}-{i}-'
                _, frame = video.read() 
                faces, _ = cv.detect_face(frame)
                for j, (x0,y0,x1,y1) in enumerate(faces):
                    face = frame[y0:y1, x0:x1]
                    face = cv2.resize(face, dimensions, interpolation = cv2.INTER_AREA)
                    filename = f'{name}face{j+1}.jpg'
                    cv2.imwrite(filename, face)
            video.release() 
            cv2.destroyAllWindows()

            #Print some data of the progress and time left
            time_spent = time()-start_time
            avg_time = (avg_time*(c)+time_spent)/(c+1)
            remaining = len(videos)-c
            print(f'{c+1}/{len(videos)} processed. {round(avg_time*remaining,2)}s left.', end='\r')
            processed += 1
            if processed >= count:
                break
            

    
preprocess_videos(count=2)