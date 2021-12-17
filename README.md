# CV_Project


MediaPipe.py: First step of image processing. All videos from the dataset are loaded, divided into single frames, cut around the face and then recomposed to create a cropped video. 

Cut_Videos_Embedding.py: After being cut, each video is processed by the Inflated 3D Convnet model obtaining a 400-dim embedded data.

orig_cropped_videos_features_0_999.pkl: file containing embedded data of all 1000 original videos.

Orig_video_NF_keras.py: Final step, where "orig_cropped_videos_features_0_999.pkl" is used as input of the Network, after being normalized, to learn the data distribution.  

Cropped_Original: folder contaning all original videos after being cut, with the only face detected and selected.


TO BE DONE: Use the trained Network with embedded manipulated videos (with 4 different methods) to detect OOD data.  
