##!/usr/bin/env python
# coding: utf-8

#@title Import the necessary modules
# TensorFlow and TF-Hub modules.
from absl import logging
from six import text_type

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed

logging.set_verbosity(logging.ERROR)

# Some modules to help with reading the FACE dataset.
import os
import tempfile
import ssl
import cv2
import numpy as np
import pickle

# Some modules to display an animation using imageio.
import imageio


import numpy as np

actions = {
    "basketball": "shooting", "biking":"biking", "diving":"diving", "golf_swing":"golf", "horse_riding":"riding",
    "soccer_juggling":"juggle", "swing":"swing", "tennis_swing":"tennis", "trampoline_jumping":"jumping",
    "volleyball_spiking":"spiking", "walking":"walk_dog"}

#subfolder = v_ + actions + _/v_0 + 1-25
# Utilities to fetch videos
ACTION_ROOT = "./UCF11_updated_mpg/"
_VIDEO_LIST = None

PATH = []

# As of July 2020, crcv.ucf.edu doesn't use a certificate accepted by the
# default Colab environment anymore.
unverified_context = ssl._create_unverified_context()

def list_action_videos():
  """Lists videos available in Face dataset."""
  global _VIDEO_LIST
  if not _VIDEO_LIST:
    tot_videos = []
    for action in actions:
        for i in range(1,26):
            number = str(i)
            if i < 10:
                number = "0"+str(i)
            path = ACTION_ROOT + action + "/v_" + actions[action] + "_" + number
            PATH.append(path)
            videos = os.listdir(path=path)
            tot_videos.append(videos)
        
    #_VIDEO_LIST = sorted(set(tot_videos))
  #return list(_VIDEO_LIST)
  return tot_videos


def fetch_action_video(video, idx):
    video_path = os.path.join(PATH[idx], video)
    return video_path

# Utilities to open video files using CV2
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0


# Use CPU instaead of GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Loading the pretrained model
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']


def predict(sample_video):
  # Add a batch axis to the sample video.
  model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]
 
  logits = i3d(model_input)['default'][0]
  return logits   
        


print("Start")

# Get the list of videos in the dataset.
action_videos = list_action_videos()


# Obtaining embedded data for each video
logits = [] 

#problem with 125
for i in range(201, len(action_videos)):
#for i in range(len(action_videos)):
    for j in range(len(action_videos[i])):
        video_path = fetch_action_video(action_videos[i][j], i)
        sample_video = load_video(video_path)
        logits.append(predict(sample_video))
        print(j)
    print(i)

  
# Saving the result file with all embedded data    
file_name = "action_videos_features_201_274.pkl"



open_file = open(file_name, "wb")
pickle.dump(logits, open_file)
open_file.close()


print("End")

