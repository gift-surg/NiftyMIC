## \file MyImageProcessingFunctions.py
#  \brief Image processing functions
# 
#  \author Michael Ebner
#  \date August 2015


## Import libraries 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import nibabel as nib           # nifti files

## Import other py-files within src-folder
from SliceStack import *

#  \param[in] -img image to rotate
#  \param[in] -ind slice index
#  \param[in] -deg rotation in degree applied counterclockwise to image
def rotate_2D_image_in_plane(img, ind, deg):


    affine = img.get_affine()
    trafo = np.identity(4)

    rotation_angle_x = 0
    rotation_angle_y = 0
    rotation_angle_z = deg*np.pi/180

    ## Rotation matrix around x-axes:
    rotation_x = np.identity(3)
    rotation_x[1,1] = np.cos(rotation_angle_x)
    rotation_x[1,2] = np.sin(rotation_angle_x)
    rotation_x[2,1] = -np.sin(rotation_angle_x)
    rotation_x[2,2] = np.cos(rotation_angle_x)

    ## Rotation matrix around y-axes:
    rotation_y = np.identity(3)
    rotation_y[0,0] = np.cos(rotation_angle_y)
    rotation_y[0,2] = -np.sin(rotation_angle_y)
    rotation_y[2,0] = np.sin(rotation_angle_y)
    rotation_y[2,2] = np.cos(rotation_angle_y)

    ## Rotation matrix around z-axes:
    rotation_z = np.identity(3)
    rotation_z[0,0] = np.cos(rotation_angle_z)
    rotation_z[0,1] = np.sin(rotation_angle_z)
    rotation_z[1,0] = -np.sin(rotation_angle_z)
    rotation_z[1,1] = np.cos(rotation_angle_z)

    ## Translation
    translation = np.array([0,0,0])

    ## Combine rotation (and translation) to affine transformation matrix
    trafo[0:3,0:3] = rotation_x.dot(rotation_y).dot(rotation_z)
    trafo[0:3,3] = translation

    ## Change of origin before applying rotation:

    ## translate by dimension/2!!
    T1 = np.array([
        [1,0,0,affine[0,2]/2],
        [0,1,0,affine[1,2]/2],
        [0,0,1,affine[2,2]/2],
        [0,0,0,1]])

    T2 = np.array([
        [1,0,0,-affine[0,2]/2],
        [0,1,0,-affine[1,2]/2],
        [0,0,1,-affine[2,2]/2],
        [0,0,0,1]])

    trafo = T1.dot(trafo).dot(T2)


    img.set_affine(np.dot(affine,trafo))

    # print affine
    # print trafo
    # print np.dot(affine,trafo)

    return img
