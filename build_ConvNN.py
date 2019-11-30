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

#tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()

    # option to input spectra_objects file (.npz)
    parser.add_argument('--spectra_objects', type=str, default=None, help='Array (.npz) that contains spectra object data')

    # parameters for convolutional layers
    parser.add_argument('--num_conv_layers', type=int, default=4, help='Number of convolutional layers to train with. Careful when setting this,\
                        the dimensionality of the image is reduced by half with each layer and will error out if there are too many!')
    parser.add_argument('--filter_size', type=int, default=32,
                        help='Number of filters in starting convolutional layer, doubles with every convolutional block')

    # parameters for dense layers
    parser.add_argument('--n_dense1', type=int, default=128, help='Number of neurons in first dense layer')
    parser.add_argument('--n_dense2', type=int, default=64, help='Number of neurons in second dense layer')

    # parameters for signal-to-noise ratio of FRB
    parser.add_argument('--weight_FRB', type=float, default=10.0, help='Weighting (> 1) on FRBs, used to minimize false negatives')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model training')
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs to train with')

    # save the model, confusion matrix for last epoch, and validation set
    parser.add_argument('--save_model', dest='best_model_file', type=str, default='./models/best_model.h5',
                        help='Filename to save best model in')
    parser.add_argument('--save_confusion_matrix', dest='conf_mat', metavar='confusion matrix name', type=str,
                        default='./confusion_matrices/confusion_matrix.png', help='Filename to store final confusion matrix')
    parser.add_argument('--save_classifications', type=str, default=None,
                        help='Where to save classification results (TP, FP, etc.) and prediction probabilities')

    args = parser.parse_args()

    # Read archive files and extract data arrays
    best_model_name = args.best_model_file  # Path and Pattern to find all the .ar files to read and train on
    confusion_matrix_name = args.conf_mat
    results_file = args.save_classifications
    spectra_objects = np.load(args.spectra_objects, allow_pickle=True)

    spectra_array = np.array(spectra_objects['spectra'])
    index = 0
    invalid_indices, dmvstime_array = [], []

    #Generate array of DM vs. Time arrays that represent each Spectra object
    for data in spectra_array:
        if data.dm < 50: #Filter out Spectra objects with low DM
            invalid_indices.append(index)
        else:
            dmvstime_array.append(create_dmvstime_array(data))
        index += 1

    dmvstime_array = np.array(dmvstime_array)

    classification_labels = spectra_objects['labels']
    classification_labels = np.array([classification_labels[i] for i in range(len(classification_labels)) if i not in invalid_indices])

    # Scale data
    median = np.array([np.median(dmvstime) for dmvstime in dmvstime_array])
    stddev = np.array([np.std(dmvstime) for dmvstime in dmvstime_array])
    dmvstime_array_scaled = (dmvstime_array - median) / stddev
    print(median)
    print(stddev)
    print(dmvstime_array_scaled[0:2])


    # 4D vector for Keras
    dmvstime_array_scaled = dmvstime_array_scaled[..., None]

    indices = np.arange(len(dmvstime_array_scaled))
    np.random.shuffle(indices)

    list_splitter_index = int(len(dmvstime_array_scaled) * 0.5)

    train_indices = indices[list_splitter_index:]
    test_indices = indices[:list_splitter_index]

    #Split data into the training set and the evaluation set (the set the model will predict)
    train_data = np.array(dmvstime_array_scaled[train_indices])
    train_labels = classification_labels[train_indices]
    eval_data = np.array(dmvstime_array_scaled[test_indices])
    eval_labels = classification_labels[test_indices]

    # Convert the classification labels to binary number representation: encode RFI as [1, 0] and FRB as [0, 1]
    train_labels_keras = to_categorical(train_labels)
    eval_labels_keras = to_categorical(eval_labels)

    NTIME = 256

    # used to enable saving the model
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    start_time = time()

    # Fit convolutional neural network to the training data
    score = construct_conv2d(train_data=train_data, train_labels=train_labels_keras,
                            eval_data=eval_data, eval_labels=eval_labels_keras, epochs=args.epochs, batch_size=args.batch_size,
                            num_conv_layers=args.num_conv_layers, filter_size=args.filter_size,
                            n_dense1=args.n_dense1, n_dense2=args.n_dense2,
                            weight_FRB=args.weight_FRB, saved_model_name=best_model_name)

    model_freq_time = load_model(best_model_name, compile=True)
    y_pred_prob = model_freq_time.predict(eval_data)[:, 1]
    y_pred_freq_time = np.round(y_pred_prob)

    print("Training on {0} samples took {1} minutes".format(len(train_labels), np.round((time() - start_time) / 60, 2)))

    # print out scores of various metrics
    accuracy, precision, recall, fscore, conf_mat = print_metric(eval_labels, y_pred_freq_time)

    TP, FP, TN, FN = get_classification_results(eval_labels, y_pred_freq_time)

    if results_file is not None:
        print("Saving classification results to {0}".format(results_file))
        np.savez(results_file, TP=TP, FP=FP, TN=TN, FN=FN, probabilities=y_pred_prob)

    # get lowest confidence selection for each category
    if TP.size:
        TPind = TP[np.argmin(y_pred_prob[TP])]  # Min probability True positive candidate
        TPdata = eval_data[..., 0][TPind]
    else:
        TPdata = np.zeros((48, NTIME))

    if FP.size:
        FPind = FP[np.argmax(y_pred_prob[FP])]  # Max probability False positive candidate
        FPdata = eval_data[..., 0][FPind]
    else:
        FPdata = np.zeros((48, NTIME))

    if FN.size:
        FNind = FN[np.argmax(y_pred_prob[FN])]  # Max probability False negative candidate
        FNdata = eval_data[..., 0][FNind]
    else:
        FNdata = np.zeros((48, NTIME))

    if TN.size:
        TNind = TN[np.argmin(y_pred_prob[TN])]  # Min probability True negative candidate
        TNdata = eval_data[..., 0][TNind]
    else:
        TNdata = np.zeros((48, NTIME))

    # plot the confusion matrix and display
    plt.subplot(221)
    plt.gca().set_title('TP: {}'.format(conf_mat[0][0]))
    plt.imshow(TPdata, aspect='auto', interpolation='none')
    plt.subplot(222)
    plt.gca().set_title('FP: {}'.format(conf_mat[0][1]))
    plt.imshow(FPdata, aspect='auto', interpolation='none')
    plt.subplot(223)
    plt.gca().set_title('FN: {}'.format(conf_mat[1][0]))
    plt.imshow(FNdata, aspect='auto', interpolation='none')
    plt.subplot(224)
    plt.gca().set_title('TN: {}'.format(conf_mat[1][1]))
    plt.imshow(TNdata, aspect='auto', interpolation='none')
    plt.tight_layout()

    # save data, show plot
    print("Saving confusion matrix to {}".format(confusion_matrix_name))
    plt.savefig(confusion_matrix_name, dpi=300)
    plt.show()
