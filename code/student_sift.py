import numpy as np 
olderr = np.seterr(all='ignore')
import cv2
import scipy
from scipy import ndimage, spatial
import math
import pylab

def img_padding(img_data, padding_number):

  # if condition: grayscale images
  if True:
    i, j= img_data.shape
    zeros_column = np.zeros((1, i))
    zeros_row =  np.zeros((1, j + 2*padding_number))
    # for loop: to add zero padding to columns
    counter = 1
    for l in range(0, padding_number):

      img_data = np.insert(img_data, 0, zeros_column, axis=1)
      img_data = np.insert(img_data, j + counter , zeros_column, axis=1)
      counter = counter + 1

    # for loop: to add zero padding to rows
    counter = 1
    for p in range(0, padding_number):

      img_data = np.insert(img_data, 0, zeros_row, axis=0)
      img_data = np.insert(img_data, i + counter , zeros_row, axis=0)
      counter = counter + 1
    return img_data

def get_Ix(inp_img):
    height, width = inp_img.shape
    h, w = inp_img.shape
    Ix = np.zeros((h-1, w-1))
    for row in range(0, h-1):
        for col in range(0, w-1):
            Ix[row, col] = inp_img[row, col + 1] - inp_img[row, col]
    return Ix

def get_Iy(inp_img):
    h, w = inp_img.shape
    Iy = np.zeros((h-1, w-1))
    for col in range(0, w-1):
        for row in range(0, h-1):
            Iy[row, col] = inp_img[row + 1, col] - inp_img[row, col]
    return Iy

def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        pixel_angles where gradient distributions will be described.
    (2) each cell should have a pixel_magintude_all of the local distribution of
        gradients in 8 orientations. Appending these pixel_magintude_alls together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation pixel_angles in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation pixel_angles within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    ############################################################################
    padding_number = int(abs(feature_width/2))
    image = img_padding(image, padding_number)
    dx = get_Ix(image)
    dy = get_Iy(image)
    features_descriptors=[]
    number_of_interest_points = x.size
    bin_val_1 = -1
    bin_val_2 = 1
    bin_val_3 = 2
    for interest_point in range(0, number_of_interest_points):
        o = int(abs(feature_width/4))
        pixel_magintude_all = np.zeros((o,o,8))

        for j in range(0, feature_width):
            for i in range(0, feature_width):
                ix_value = dx[(int)(y[interest_point]) + j, (int)(x[interest_point]) +i]
                iy_value = dy[(int)(y[interest_point]) +j, (int)(x[interest_point]) +i]
                pixel_angle = np.arctan2(iy_value, ix_value)
                if pixel_angle > bin_val_2:
                    pixel_angle = bin_val_3
                if pixel_angle < bin_val_1:
                    pixel_angle = bin_val_1            
                pixel_magintude = np.sqrt(ix_value**2 + iy_value**2)
                if ix_value >0:
                    pixel_magintude_all[(int)(j/4), (int)(i/4), math.ceil(pixel_angle+1)] += pixel_magintude
                else:
                    pixel_magintude_all[(int)(j/4), (int)(i/4), math.ceil(pixel_angle+5)] += pixel_magintude
        single_feat = np.reshape(pixel_magintude_all,(1,o*o*8))
        single_feat = single_feat/(single_feat.sum())
        features_descriptors.append(single_feat)
    fv = np.array(features_descriptors)
    #############################################################################
    #                             END OF YOUR CODE                  
    #############################################################################
    return fv


    raise NotImplementedError('`get_features` function in ' +
        '`student_sift.py` needs to be implemented')
