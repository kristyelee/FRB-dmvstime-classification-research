import numpy as np
import matplotlib.pyplot as plt
from time import time
import os, sys
from tqdm import tqdm, trange  # progress bar
import argparse  # to parse arguments in command line
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import load_model

from dmvstime_plot import *

# Create a model, and helper functions for training
from training_utils import *
from model import construct_conv2d

# import waterfaller and filterbank from Vishal's path
sys.path.append('/usr/local/lib/python2.7/dist-packages/')
sys.path.append('/home/vgajjar/linux64_bin/lib/python2.7/site-packages/')

# generate Spectra objects for FRB injection
from waterfaller import filterbank, waterfall
import copy

# Utility Function: Plots 5 dispersed signals and DM vs time plots for 5 Spectra objects containing FRBs

def generate_five(spectra_array, classification_labels):
    count = 0

    for i in range(len(spectra_array)):
        if classification_labels[i] == 1:
            print(i, classification_labels[i])
            print(spectra_array[i].freqs.max(), spectra_array[i].freqs.min(), spectra_array[i].dm)
            dmvstm_array = create_dmvstime_array(spectra_array[i])
            spectra_array[i].dm = 0
            plt.imshow(np.fliplr(spectra_array[i].data), aspect='auto', extent=(0, 250, spectra_array[i].freqs.min(), spectra_array[i].freqs.max()))
            plt.show()
            plt.imshow(dmvstm_array, aspect='auto')
            plt.show()
            count += 1
            if count == 5:
                break
        else:
            continue



if __name__=='__main__':
    # Command Line Generation of Spectra Object
    spectras = np.load(sys.argv[1], allow_pickle=True)
    spectra_array = np.array(spectras['spectra'])
    classification_labels = spectras['labels']

    generate_five(spectra_array, classification_labels)
