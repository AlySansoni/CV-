from absl import logging


logging.set_verbosity(logging.ERROR)

import numpy as np
import pickle


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt


dict_names = ["orig", "DeepFake", "Face2Face", "FaceSwap", "NeuralTextures" ]
log_likelihood = []

#Open all computed data
for name in dict_names:
    file_name = name+"_log_likelihood.pkl"
    open_file_original = open(file_name, "rb")
    log_likelihood.append(pickle.load(open_file_original))
    open_file_original.close()

#Plotting the results
f, axes = plt.subplots(3, 2)
f.set_size_inches(40, 30)
f.suptitle("LOG-Likelihood Histograms")

axes[0, 0].hist(np.array(log_likelihood[0]), color = 'blue')
axes[0, 0].set(title="Original", xlabel="Log-Likelihood")
axes[0, 1].hist(np.array(log_likelihood[1]), color = 'yellow')
axes[0, 1].set(title="DeepFakes", xlabel="Log-Likelihood")
axes[1, 0].hist(np.array(log_likelihood[2]), color = 'green')
axes[1, 0].set(title="Face2Face", xlabel="Log-Likelihood")
axes[1, 1].hist(np.array(log_likelihood[3]), color = 'red')
axes[1, 1].set(title="FaceSwap", xlabel="Log-Likelihood")
axes[2, 0].hist(np.array(log_likelihood[4]), color = 'purple')
axes[2, 0].set(title="NeuralTextures", xlabel="Log-Likelihood")

axes[2,1].hist(np.array(log_likelihood[0]), alpha = 1 ,label='Original', histtype = 'step')
axes[2,1].hist(np.array(log_likelihood[1]), alpha = 0.7, label='DeepFakes', histtype = 'step')
axes[2,1].hist(np.array(log_likelihood[2]), alpha = 0.7, label='Face2Face', histtype = 'step')
axes[2,1].hist(np.array(log_likelihood[3]), alpha = 0.7, label='FaceSwap', histtype = 'step')
axes[2,1].hist(np.array(log_likelihood[4]), alpha = 0.7, label='NeuralTextures', histtype = 'step')
axes[2,1].set(title ='Log-likelihood distribution', xlabel="Log-Likelihood")

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

plt.legend()
plt.show()
