'''
Command-line interface for running the classifier on a video from a local path or a youtube video by url
'''

import argparse
from interface import predict_video, predict_youtube

parser = argparse.ArgumentParser(description='Determine whether a video is real or fake.')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--path','-p', action='store', type=str, help='The path to the video to be classified.')
group.add_argument('--url', '-u', action='store', type=str, help='The url to a youtube video to be classified.')
args = parser.parse_args()

path = args.path
url = args.url

if path:
    predict_video(path, printing=True)
elif url:
    predict_youtube(url, printing=True)
