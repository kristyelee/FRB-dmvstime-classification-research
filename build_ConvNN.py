import numpy as np
import matplotlib.pyplot as plt
from time import time
import os, sys
from tqdm import tqdm, trange  # progress bar
import argparse  # to parse arguments in command line
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import load_model

# simulate FRB, create a model, and helper functions for training
from simulate_FRB import SimulatedFRB
from training_utils import *
from model import construct_conv2d

# import waterfaller and filterbank from Vishal's path
sys.path.append('/usr/local/lib/python2.7/dist-packages/')
sys.path.append('/home/vgajjar/linux64_bin/lib/python2.7/site-packages/')

# generate Spectra objects for FRB injection
from waterfaller import filterbank, waterfall
import copy

"""Adapted from the code published alongside the paper 'Applying Deep Learning
to Fast Radio Burst Classification' by Liam Connor and Joeri van Leeuwen, as
well as code wrapping done by Vishal Gajjar. Consulted code written by Dominic
Dleduc for FRB classification."""

"""Trains a convolutional neural network to recognize differences between fast
radio bursts and RFI. Training is done by generating a DM versus time plot for
each image, storing data of the plot into a numpy array, and using existing
classification information of whether that numpy array for each image corresponds
to the presence of an FRB. Pass in an .npz file that contains 1) an array of Spectra
objects, and 2) classification labels corresponding to whether the Spectra object
contains an FRB or not. With the array of Spectra objects, convert each into a
DM vs time plot, then match each DM vs time plot to classification labels through
indexing, and finally train the convolutional neural network through classification
of the DM vs time plot as indicative of containing an FRB or just RFI."""
