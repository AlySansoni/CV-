# CV_Project


Link to download the Dataset with relative instructions: https://github.com/ondyari/FaceForensics

Images folder: contains data deriving from different computation steps

src folder: contains the scripts of all the stages of the process

Data sub-folder: contains the file orig_cropped_videos_features_0_999.pkl with embedded data of all 1000 original videos.

MediaPipe.py: First step of image processing. All videos from the dataset are loaded, divided into single frames, cut around the face and then recomposed to create a cropped video. 

Cut_Videos_Embedding.py: After being cut, each video is processed by the Inflated 3D Convnet model obtaining a 400-dim embedded data.

Orig_video_NF_keras.py: Final step of the training, where "orig_cropped_videos_features_0_999.pkl" is used as input of the Network, after being normalized, to learn the data distribution.  

Man_Video_NF_test.py: Script to load and test manipulated video using the Net trained at the previous step

Log_likelihood.py: Script to compute and plot all the log-likelihoods. 

ActionVideos_Embedding.py: Script to obtain the embeddings for action dataset.
