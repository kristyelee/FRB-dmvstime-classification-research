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
radio bursts and RFI. Training is done by simulating a specified number of FRB
examples and injecting them into noisy Gaussian backgrounds. To include actual
RFI data, extract_spectra gets real data from filterbank files and turns them
into numpy arrays that this program can inject FRBs into."""





