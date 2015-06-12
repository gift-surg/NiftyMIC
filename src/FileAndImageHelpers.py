## \file FileAndImageHelpers.py
#  \brief Implementation of some helpful functions related to images
# 
#  \author Michael Ebner
#  \date May 2015


## Import libraries 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import nibabel as nib           # nifti files


def read_file(filename):
    imagefile = Image.open(filename)
    image_array = np.array(imagefile.getdata(), 
        np.uint8).reshape(imagefile.size[1], imagefile.size[0])
    return image_array.astype('float32')


def array_to_image(array):
    minimal_value = np.min(array)
    maximal_value = np.max(array)

    if minimal_value < 0 or maximal_value > 255:
        array = 255 * (array - minimal_value) / float(maximal_value - minimal_value)

    array_uint8 = array.astype('uint8')
    return Image.fromarray(array_uint8, 'L')

def save_file(array, filename):
    imagefile = array_to_image(array)
    imagefile.save(filename)
    return None


def display_image(array):
    png = array_to_image(array)
    png.format = "PNG"
    png.show()
    return None


## Function to display row of image slices
#  (http://nipy.org/nibabel/coordinate_systems.html)
#  Must be more than 1 slice!!
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def normalize_image(array):
    minimal_value = np.min(array)
    maximal_value = np.max(array)
    return (array - minimal_value) / float(maximal_value - minimal_value)