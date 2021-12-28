#!/usr/bin/env python
# coding: utf-8

#@title Import the necessary modules

from absl import logging
from matplotlib import axes


logging.set_verbosity(logging.ERROR)

# Some modules to help with reading the FACE dataset.

import numpy as np
import pickle


# Some modules to display an animation using imageio.


from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

# As of July 2020, crcv.ucf.edu doesn't use a certificate accepted by the
# default Colab environment anymore.

#Loading original data 
dict_names = ["orig", "DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures" ]
loaded_list = []
norm_data_list = []

for name in dict_names:
    file_name = name+"_cropped_videos_features_0_999.pkl"
    open_file_original = open(file_name, "rb")
    loaded_list.append(pickle.load(open_file_original))
    open_file_original.close()

for i in range(len(loaded_list)):
    data = np.array(loaded_list[i])
    # Data Normalization
    norm = layers.Normalization()
    norm.adapt(data)
    norm_data_list.append(norm(data))


# Creating a custom layer with keras API.

model = keras.models.load_model('./Model')


orig_data, logdet_orig = model(norm_data_list[0], training = False)

plt.figure(figsize=(15, 10))
plt.scatter(orig_data[:,0],orig_data[:,1])
#plt.xlim([-5, 5])


deepFake_data, logdet_orig = model(norm_data_list[1], training = False)
face2Face_data, logdet_orig = model(norm_data_list[2], training = False)
faceSwap_data, logdet_orig = model(norm_data_list[3], training = False)
neuralTextures_data, logdet_orig = model(norm_data_list[4], training = False)

f, axes = plt.subplots(2, 2)
f.set_size_inches(20, 15)



#axes[0, 0].hist(deepFake_data)
axes[0, 0].scatter(deepFake_data[:,0], deepFake_data[:,1])
axes[0, 0].set(title="deepFake", xlabel="data", ylabel="logdet")
#axes[0, 0].set_xlim([-5, 5])
#axes[0, 1].hist(face2Face_data)
axes[0, 1].scatter(face2Face_data[:,0], face2Face_data[:,1])
axes[0, 1].set(title="face2Face", xlabel="data", ylabel="logdet")
#axes[0, 1].set_xlim([-5, 5])
#axes[1, 0].hist(faceSwap_data)
axes[1, 0].scatter(faceSwap_data[:,0], faceSwap_data[:,1])
axes[1, 0].set(title="faceSwap", xlabel="data", ylabel="logdet")
#axes[1, 0].set_xlim([-5, 5])
#axes[1, 1].hist(neuralTextures_data)
axes[1, 1].scatter(neuralTextures_data[:,0], neuralTextures_data[:,1])
axes[1, 1].set(title="neuralTextures", label="x", ylabel="y")
#axes[1, 1].set_xlim([-5, 5])


# From data to latent space.
#z, _ = model(normalized_data)


# From latent space to data.
#samples = model.distribution.sample(3000)
#print(type(samples))

#x, logdet = model.predict(samples)


#print(y)
#plt.hist(x)
#print(z)
#print(type(z))
#print(logdet)
#log_likelihood = z.distribution.log_prob(z) + logdet
#print(log_likelihood)

"""
plt.figure(figsize=(15, 10))
plt.plot(log_likelihood["LogLikelihood"])
plt.title("model likelihood")
#plt.ylabel("loss")
#plt.xlabel("epoch")




f, axes = plt.subplots(2, 2)
f.set_size_inches(20, 15)



axes[0, 0].scatter(normalized_data[:, 0], normalized_data[:, 1], color="r")
axes[0, 0].set(title="Inference data space X", xlabel="x", ylabel="y")
axes[0, 1].scatter(fake_res[:, 0], fake_res[:, 1], color="r")
axes[0, 1].set(title="Inference latent space Z", xlabel="x", ylabel="y")
axes[0, 1].set_xlim([-3.5, 4])
axes[0, 1].set_ylim([-4, 4])
axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g")
axes[1, 0].set(title="Generated latent space Z", xlabel="x", ylabel="y")
axes[1, 1].scatter(x[:, 0], x[:, 1], color="g")
axes[1, 1].set(title="Generated data space X", label="x", ylabel="y")
axes[1, 1].set_xlim([-3, 3])
axes[1, 1].set_ylim([-3, 3])

"""



plt.show()
plt.waitforbuttonpress()
