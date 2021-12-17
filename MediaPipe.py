#!/usr/bin/env python
# coding: utf-8

#@title Import the necessary modules
# TensorFlow and TF-Hub modules.
from absl import logging

import math

logging.set_verbosity(logging.ERROR)

# Some modules to help with reading the FACE dataset.
import os

import ssl
import cv2

import mediapipe as mp


#@title Helper functions

# Utilities to fetch videos
#FACE_ROOT = "./original_sequences/youtube/c40/videos"
FACE_ROOT = "./manipulated_sequences/NeuralTextures/c40/videos"
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


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_videos = list_face_videos()

for j in range(len(face_videos)): # loop for all videos in the folder

  video_path = fetch_face_video(face_videos[j])

  vidcap = cv2.VideoCapture(video_path)

  video_array = [] #to save all frames
  cut_video_array = [] #to save all cut frames
  det_faces = [] #to save coordinates of detected bounding boxes
    
  # face detection for each frame of a single  
  with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    count = 0
    success = True

    while success:
      success, image = vidcap.read()
      if (not success):
        break

      
      # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      # Draw face detections of each face.
      if not results.detections:
        continue
      annotated_image = image.copy()
      
      faces = []
    
      for detection in results.detections:
        faces.append(detection.location_data.relative_bounding_box)
      
      # Heuristic to chose the face with the biggest area if in the first frame there is more than one
      if count == 0:
        if len(faces) > 1:
          if faces[0].width*faces[0].height >= faces[1].width*faces[1].height :
            faces.pop(1)
          else:
            faces.pop(0)

        # Setting the first selected face as reference for the others
        ref_frame = faces[0]
    

      # Heuristic to chose the most coherent face wrt to the reference one if there is more than one
      if len(faces) > 1:
        if abs(ref_frame.xmin - faces[0].xmin) < abs(ref_frame.xmin - faces[1].xmin):
          faces.pop(1)
        else:
          faces.pop(0)
    
      det_faces.append(faces)
      video_array.append(annotated_image)
    
      count += 1


    # Obtaining the index of the frame with max Area Boundind Box
    ind_max_dim = max(enumerate(det_faces), key=lambda x:x[1][0].width*x[1][0].height)[0]

    image_height, image_width, _ = video_array[0].shape 
    
    width_measure = math.ceil((det_faces[ind_max_dim][0].width)*image_width)
    height_measure = math.ceil((det_faces[ind_max_dim][0].height)*image_height)

    # Cutting all frames with the max detected box dimension
    for idx in range(len(video_array)):
      x_coord = math.ceil((det_faces[idx][0].xmin)*image_width)
      y_coord = math.ceil((det_faces[idx][0].ymin)*image_height)
      cut_video_array.append(video_array[idx][y_coord:(y_coord+height_measure), x_coord:(x_coord+width_measure)])
    
    """
    # If frame dimensions are not all the same
    min_dim = len(cut_video_array[0])
    for bab in range(len(cut_video_array)):
      curr_dim = len(cut_video_array[bab])
      if curr_dim < min_dim:
        min_dim = curr_dim

    new_cut_video_array = []
    for idx2 in range (len(cut_video_array)):
      new_cut_video_array.append(cut_video_array[idx2][0:min_dim, 0:min_dim])

    size = (min_dim, min_dim)"""

    size = (height_measure,width_measure)


    # Writing resulting cropped video.
    out = cv2.VideoWriter('Cropped_NeuralTextures/' +str(j) +'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(len(cut_video_array)):
      out.write(cut_video_array[i])
    out.release()
    
print("Done")
