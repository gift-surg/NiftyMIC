#!/usr/bin/python

## \file main.py
#  \brief main-file incorporating all the other files 
# 
#  \author Michael Ebner
#  \date June 2015


## Import libraries 
import os                       # used to execute terminal commands in python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import nibabel as nib           # nifti files


## Import other py-files within src-folder
from FetalImage import *
from FileAndImageHelpers import *
from Registration import *
# from SegmentationPropagation import *
# from SimilarityMeasures import *
# from StatisticalAnalysis import *


""" ###########################################################################
    main:
"""
def main():
    
    # dir_input = "/Users/mebner/development/BTK/meb/FetalNeckImages/R2/data_cropped/"
    dir_input = "../data/"
    filename = [
        "20150115_161038s003a1001_crop",
        "20150115_161038s004a1001_crop",
        "20150115_161038s005a1001_crop",
        "20150115_161038s006a1001_crop",
        "20150115_161038s007a1001_crop",
        "20150115_161038s5005a1001_crop",
        "20150115_161038s5006a1001_crop",
        "20150115_161038s5007a1001_crop"
        ]

    N = len(filename)
    target_stack_id = 3

    fetal_stacks = []

    for i in range(0,N):
        fetal_stacks.append(FetalImage(dir_input, filename[i]))


    registration = Registration(fetal_stacks, target_stack_id)
    registration.register_images()

    target = registration.get_target_stack()
    floatings = registration.get_floating_stacks()
    registered_stacks = registration.get_registered_stacks()

    print(len(registered_stacks))
    for i in range(0,len(registered_stacks)):
        print(registered_stacks[i].get_filename())
        print(registered_stacks[i].get_dir())
   


    # plt.imshow(fetal_stacks[0].getData()[:,:,10], cmap="Greys_r")
    # plt.show()



    """ #######################################################################
        Playground:
    """
    # print "Loading data"
    # imgData = read_file("../longitudinal_images/AD_0_baseline.nii.gz")

    # dir_subjects = "../longitudinal_images/"
    # dir_computedData = "tmp/be0.01_sx-5/"
    # dir_out_midspace = "tmp/midspace/"
    # image_string = "AD_0"

    # BSI = compute_BSI(dir_out_midspace, image_string)
    # V1 = compute_segmentation_volume(dir_out_midspace, image_string + "_baseline_brain_midspace")
    # V2 = compute_segmentation_volume(dir_out_midspace,  image_string + "_followup_brain_midspace")

    # print("Subject: " + image_string)
    # print("BSI = " + str(BSI))
    # print("DeltaV = " + str(V1-V2))

    # img0_nifti = nib.load(dir_input+filename)
    # img1_nifti = nib.load(dir_subjects + image_string + "_followup.nii.gz")

    # img0 = img1

    # img0 = img0_nifti.get_data()
    # img1 = img1_nifti.get_data()

    # print img0.shape
    # print img0.shape[0]
    # print range(img0.shape[0])
    # for i in xrange(2):
    #     print i


    # img0 = normalize_image(img0)
    # img1 = normalize_image(img1)

    # plot_image_overview(img0_data, img1_data)
    # plot_joint_histogram(img0_data, img1_data)

    # print img0_nifti.header
    # pixdim = img0_nifti.header.get_zooms()
    # print pixdim
    # print pixdim[0]
    # print data.shape
    # display_image(img_data[:,62,:])
    # plt.imshow(fetal_stacks[0].getData()[:,:,10], cmap="Greys_r")
    # plt.show()

    # subject_nr = 8
    # template_nr = 0


    # img0_nifti = nib.load("../longitudinal_images/AD_10_baseline_brain_template_0.nii.gz")
    # img1_nifti = nib.load("../longitudinal_images/AD_10_baseline_brain_template_8.nii.gz")
    # print("Dice = " + str(dice(img0_nifti.get_data(), img1_nifti.get_data())))
    #
    # average = np.round((img0_nifti.get_data() + img1_nifti.get_data())/2)
    # # display_image(weighted_average[:,4,:])


    # hist, bin_edges = np.histogram(img_data,100)
    # hist, x_edges, y_edges = np.histogram2d(img_data.reshape(N), img_data.reshape(N))
    # prob = hist/float(np.sum(hist))

    # print(set(img_data))

    # plt.show()


if __name__ == "__main__":
    main()
