## \file SimilarityMeasures.py
#  \brief Implementation of used similarity measures 
# 
#  \author Michael Ebner
#  \date May 2015


## Import libraries 
import os                       # used to execute terminal commands in python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import nibabel as nib           # nifti files

## Normalize array between 0 and 1
def normalize_image(array):
    minimal_value = np.min(array)
    maximal_value = np.max(array)
    return (array - minimal_value) / float(maximal_value - minimal_value)


## Sum square difference
def ssd(img1, img2):
    return np.sum(np.square(img1 - img2))


## Mean square difference
def msd(img1, img2):
    return np.sum(np.square(img1 - img2))/img1.size


## l1-norm of gradient
def grad_l1(img):
    dim = 3  # = np.array(img.shape).size
    grad = np.zeros(np.append([img.shape[i] for i in range(0, dim)], dim))
    grad_list = np.gradient(img)

    for i in range(0, dim):
        grad[:, :, :, i] = grad_list[i]

    return np.sum(np.abs(grad), dim)


## Gradient of sum squared difference
def grad_ssd(img_ref, img_warp):
    dim = 3  # = np.array(img.shape).size
    grad = np.zeros(np.append([img_warp.shape[i] for i in range(0, dim)], dim))
    ssd_grad = np.zeros(np.append([img_warp.shape[i] for i in range(0, dim)], dim))
    grad_list = np.gradient(img_warp)

    for i in range(0, dim):
        ssd_grad[:, :, :, i] = -2 * (img_ref - img_warp) * grad_list[i]

    return ssd_grad


## l1-norm of ssd-gradient
def grad_ssd_l1(img_ref, img_warp):
    return np.sum(np.abs(grad_ssd(img_ref, img_warp)), 3)


## Shannon entropy:
def entropy(img):
    bins = 100
    hist, bin_edges = np.histogram(img, bins)
    hist_length = np.sum(hist)
    probability = hist / float(np.sum(hist))
    return - sum([p * np.log2(p) for p in probability.reshape(probability.size) if p != 0])


## Shannon entropy for a joint distribution
def joint_entropy(img1, img2):
    bins = 100

    N = img1.size
    hist, x_edges, y_edges = np.histogram2d(img1.reshape(N), img2.reshape(N), bins)
    probability = hist / float(np.sum(hist))
    return - sum([p * np.log2(p) for p in probability.reshape(probability.size) if p != 0])


## Normalised cross correlation
def ncc(img1, img2):
    mean1 = np.mean(img1)
    std1 = np.std(img1)
    mean2 = np.mean(img2)
    std2 = np.std(img2)
    return np.sum((img1 - mean1) * (img2 - mean2)) / (img1.size * std1 * std2)


## Mutual information
def mi(img1, img2):
    return entropy(img1) + entropy(img2) - joint_entropy(img1, img2)


## Normalised mutual information
def nmi(img1, img2):
    return (entropy(img1) + entropy(img2)) / joint_entropy(img1, img2)


## Dice score (Dice's Score or Dice's Coefficient)
def dice(img0, img1):
    numerator = 2 * np.sum(img0 * img1)
    denominator = np.sum(img0 > 0) + np.sum(img1 > 0)
    return numerator / float(denominator)


## Joint histogram
def plot_joint_histogram(img0, img1):

    img0 = normalize_image(img0)
    img1 = normalize_image(img1)

    N = img0.size
    hist, x_edges, y_edges = np.histogram2d(img0.reshape(N), img1.reshape(N), bins=100)
    prob = hist / float(np.sum(hist))

    plt.clf()
    # plt.ion()                 # combination plt.ion() + plt.show() does not pause during compilation

    plt.figure(1)
    plt.imshow(np.ma.log(prob), origin="low", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
               interpolation="none")
    plt.colorbar()
    plt.xlabel("Reference Image")
    plt.ylabel("Warped image")
    
    # plt.show()                # execution of could would pause here
    plt.show(block=False)       # does not pause, but needs plt.show() at end 
                                # of file to be visible
    return None


## Plot of reference and warped image with similarity measure
def plot_comparison_of_reference_and_warped_image(img0, img1):
    SSD = ssd(img0, img1)
    NCC = ncc(img0, img1)
    NMI = nmi(img0, img1)
    JE = joint_entropy(img0, img1)

    # print("entropy = " + str(entropy(img0)))
    print("\nJoint Entropy = " + str(JE))
    print("SSD = " + str(SSD))
    print("NCC = " + str(NCC))
    # print("MI = " + str(mi(img0,img1)))
    print("NMI = " + str(NMI))

    y_slice = 50

    plt.figure(2)
    plt.suptitle("SSD = " + str(SSD) + "\nNCC = " + str(NCC) + "\nNMI = " + str(NMI))

    plt.subplot(121)
    plt.imshow(img0, cmap="Greys_r", origin="low")
    plt.title('Reference Image')

    plt.subplot(122)
    plt.imshow(img1, cmap="Greys_r", origin="low")
    plt.title('Warped Image')

    # plt.subplot(223)
    # plt.imshow(np.abs(img0 - img1), cmap="Greys_r")
    # plt.title('absolute difference')

    # plt.subplot(224)
    # # plt.imshow(grad_l1(img0-img1)[:,y_slice,:],cmap="Greys_r")
    # plt.imshow(grad_ssd_l1(img0, img1), cmap="Greys_r")
    # plt.title('grad_ssd')

    # plt.show()                # execution of could would pause here
    plt.show(block=False)       # does not pause, but needs plt.show() at end 
                                # of file to be visible
    return None


def my_log(x):
    return map(np.log,x)
    # for i in x:
    # if x == 0:
    #     return float("-Inf")
    # else:
    #     return np.log(abs(x))