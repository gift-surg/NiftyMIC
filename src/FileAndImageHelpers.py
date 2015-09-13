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

## Import other py-files within src-folder
from SimilarityMeasures import *


def read_file(filename):
    imagefile = Image.open(filename)
    image_array = np.array(imagefile.getdata(), np.uint8)


    if len(image_array.shape)==1:
        image_array = image_array.reshape(imagefile.size[1], imagefile.size[0])
    else:
        image_array = image_array[:,0].reshape(imagefile.size[1], imagefile.size[0])

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


## Plot of reference and warped image with similarity measure
def plot_images(images, titles, display_output=0, fig_id=-1):

    N = images.shape[-1]

    if fig_id == -1:
        fig = plt.figure()
    else:
        fig = plt.figure(fig_id)
        plt.clf()

    if display_output:
        print("2")

    plt.subplot(121)
    # plt.imshow(fixed, cmap="Greys_r", origin="low")
    plt.imshow(fixed, cmap="Greys_r")
    plt.title(fixed_title)

    plt.subplot(122)
    # plt.imshow(warped, cmap="Greys_r", origin="low")
    plt.imshow(warped, cmap="Greys_r")
    plt.title(warped_title)

    # plt.subplot(223)
    # plt.imshow(np.abs(fixed - warped), cmap="Greys_r")
    # plt.title('absolute difference')

    # plt.subplot(224)
    # # plt.imshow(grad_l1(fixed-warped)[:,y_slice,:],cmap="Greys_r")
    # plt.imshow(grad_ssd_l1(fixed, warped), cmap="Greys_r")
    # plt.title('grad_ssd')

    # plt.show()                # execution of code would pause here
    plt.show(block=False)       # does not pause, but needs plt.show() at end 
                                # of file to be visible
    return fig


## Plot of reference and warped image with similarity measure
def plot_comparison_of_reference_and_warped_image(fixed, warped, fig_id=-1, display_output=0, 
    fixed_title="Reference Image", warped_title="Warped Image"):

    # y_slice = 50

    if fig_id == -1:
        fig = plt.figure()
    else:
        fig = plt.figure(fig_id)
        plt.clf()

    if display_output:
        SSD = ssd(fixed, warped)
        NCC = ncc(fixed, warped)
        NMI = nmi(fixed, warped)
        JE = joint_entropy(fixed, warped)
        
        # print("entropy = " + str(entropy(fixed)))
        print("\nJoint Entropy = " + str(JE))
        print("SSD = " + str(SSD))
        print("NCC = " + str(NCC))
        # print("MI = " + str(mi(fixed,warped)))
        print("NMI = " + str(NMI))

        plt.suptitle("SSD = " + str(SSD) + "\nNCC = " + str(NCC) + "\nNMI = " + str(NMI))


    plt.subplot(121)
    # plt.imshow(fixed, cmap="Greys_r", origin="low")
    plt.imshow(fixed, cmap="Greys_r")
    plt.title(fixed_title)

    plt.subplot(122)
    # plt.imshow(warped, cmap="Greys_r", origin="low")
    plt.imshow(warped, cmap="Greys_r")
    plt.title(warped_title)

    # plt.subplot(223)
    # plt.imshow(np.abs(fixed - warped), cmap="Greys_r")
    # plt.title('absolute difference')

    # plt.subplot(224)
    # # plt.imshow(grad_l1(fixed-warped)[:,y_slice,:],cmap="Greys_r")
    # plt.imshow(grad_ssd_l1(fixed, warped), cmap="Greys_r")
    # plt.title('grad_ssd')

    # plt.show()                # execution of code would pause here
    plt.show(block=False)       # does not pause, but needs plt.show() at end 
                                # of file to be visible
    return fig