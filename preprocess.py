import os
import cv2
import random
import pathlib
from time import time

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


def preprocess_videos(dimensions=(500,500), sample=5):
    '''
    Runs through all the videos downloaded and saves 'samples' amount of frames with 'dimensions' dimensions.

    Returns
        None
    '''
    processed_path = create_processed_dirs()
    for label, label_path in datasets.items():  # Loop through labels (real, fake) with their corresponding dataset paths
        path = current_dir + data_dir + label_path
        videos = os.listdir(path)
        avg_time = 0
        for c, video_name in enumerate(videos):
            start_time = time()
            video_path = path + video_name
            video = cv2.VideoCapture(video_path)
            image_path = processed_path + label + '/' + video_name.split('.')[0]

            # Saves frames from a random sampled subset of the video
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            sampled_frames = random.sample([i for i in range(frame_count)], sample) 
            for i in sampled_frames:
                video.set(1,i)
                name = f'{image_path}-{i}.jpg'
                _, frame = video.read() 
                frame = cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)
                cv2.imwrite(name, frame)
            video.release() 
            cv2.destroyAllWindows()

            #Print some data of the progress and time left
            time_spent = time()-start_time
            avg_time = (avg_time*(c)+time_spent)/(c+1)
            remaining = len(videos)-c
            print(f'{c+1}/{len(videos)} processed. {round(avg_time*remaining,2)}s left.', end='\r')

    
preprocess_videos()