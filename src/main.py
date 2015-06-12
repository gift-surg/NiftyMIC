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

    for i in range(0,len(registered_stacks)):
        print registered_stacks[i].get_filename()





    # print(len(registered_stacks))
    # for i in range(0,len(registered_stacks)):
    #     print(registered_stacks[i].get_filename())
    #     print(registered_stacks[i].get_dir())
   

    # display_image(registration.get_target_stack().get_data()[:,:,10])

    # plt.imshow(registration.get_target_stack().get_data()[:,:,10], cmap="Greys_r")

    img = registration.get_HR_stack().get_data()
    N = img.shape

    slice_0 = img[:,:,np.round(N[2]/2)]
    slice_1 = img[:,np.round(N[1]/2),:]
    slice_2 = img[np.round(N[0]/2),:,:]

    show_slices([slice_0,slice_1,slice_2])
    plt.show()



    """ #######################################################################
        Playground:
    """
    nifti = nib.load(dir_input+filename[3]+".nii.gz")

    # print nib.aff2axcodes(nifti.affine)
    # print nib.zooms

if __name__ == "__main__":
    main()
