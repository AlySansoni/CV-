##!/usr/bin/env python
# coding: utf-8

#@title Import the necessary modules
# TensorFlow and TF-Hub modules.
from absl import logging

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


# Utilities to fetch videos
FACE_ROOT = "./Cropped_Original"
_VIDEO_LIST = None

# As of July 2020, crcv.ucf.edu doesn't use a certificate accepted by the
# default Colab environment anymore.
unverified_context = ssl._create_unverified_context()

def list_face_videos():
  """Lists videos available in Face dataset."""
  global _VIDEO_LIST
  if not _VIDEO_LIST:
    videos = os.listdir(path=FACE_ROOT)
    _VIDEO_LIST = sorted(set(videos))
  return list(_VIDEO_LIST)


def fetch_face_video(video):
    video_path = os.path.join(FACE_ROOT, video)
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

def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=25)
  return embed.embed_file('./animation.gif')

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
face_videos = list_face_videos()

# Obtaining embedded data for each video
logits = [] 
for i in range(1000):
    name = face_videos[i]
    video_path = fetch_face_video(face_videos[i])
    sample_video = load_video(video_path)
    logits.append(predict(sample_video))
    print(i)

  
# Saving the result file with all embedded data    
file_name = "orig_cropped_videos_features_0_999.pkl"

open_file = open(file_name, "wb")
pickle.dump(logits, open_file)
open_file.close()


print("End")
