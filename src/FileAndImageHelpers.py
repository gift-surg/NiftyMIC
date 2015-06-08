## \file FileAndImageHelpers.py
#  \brief Implementation of some helpful functions related to images
# 
#  \author Michael Ebner
#  \date May 2015


## Import libraries 
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
        array = 255 * (array - minimal_value) / (maximal_value - minimal_value)

    array_uint8 = array.astype('uint8')
    return Image.fromarray(array_uint8, 'L')


def save_file(array, filename):
    imagefile = array_to_image(array)
    imagefile.save(filename)
    return None


def display_image(array):
    png = array_to_image(array)
    png.show()
    return None


def normalize_image(array):
    minimal_value = np.min(array)
    maximal_value = np.max(array)
    return (array - minimal_value) / float(maximal_value - minimal_value)