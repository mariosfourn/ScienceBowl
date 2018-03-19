import pandas as pd                 
import numpy as np                                       
import sklearn.model_selection     # For using KFold
import keras.preprocessing.image   # For using image generation
import datetime                    # To measure running time 
import skimage.transform           # For resizing images
import skimage.morphology          # For using image labeling
import cv2                         # To read and manipulate images
import os                          # For filepath, directory handling
import sys                         # System-specific parameters and functions
import tqdm                        # Use smart progress meter



def rle_of_binary(x):
    """ Run length encoding of a binary 2D array. """
    dots = np.where(x.T.flatten() == 1)[0] # indices from top to down
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def mask_to_rle(mask, cutoff=.5, min_object_size=1.):
    """ Return run length encoding of mask. """
    # segment image and label different objects
    lab_mask = skimage.morphology.label(mask > cutoff)
    
    # Keep only objects that are large enough.
    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
    if (mask_sizes < min_object_size).any():
        mask_labels = mask_labels[mask_sizes < min_object_size]
        for n in mask_labels:
            lab_mask[lab_mask == n] = 0
        lab_mask = skimage.morphology.label(lab_mask > cutoff) 
        
    # Loop over each object excluding the background labeled by 0.
    for i in range(1, lab_mask.max() + 1):
        yield rle_of_binary(lab_mask == i)
        
def rle_to_mask(rle, img_shape):
    ''' Return mask from run length encoding.'''
    mask_rec = np.zeros(img_shape).flatten()
    for n in range(len(rle)):
        for i in range(0,len(rle[n]),2):
            for j in range(rle[n][i+1]): 
                mask_rec[rle[n][i]-1+j] = 1
    return mask_rec.reshape(img_shape[1], img_shape[0]).T
        