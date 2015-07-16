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

import numpy as np
import numpy.linalg as npl


## Import other py-files within src-folder
from DataPreprocessing import *
from SliceStack import *
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
    dir_out = "../results/"

    ## Fetal Neck Images:
    dir_input = "../data/"
    filenames = [
        "20150115_161038s006a1001_crop",
        "20150115_161038s003a1001_crop",
        "20150115_161038s004a1001_crop",
        "20150115_161038s005a1001_crop",
        "20150115_161038s007a1001_crop",
        "20150115_161038s5005a1001_crop",
        "20150115_161038s5006a1001_crop",
        "20150115_161038s5007a1001_crop"
        ]

    ## Kidney Images:
    dir_input = "/Users/mebner/UCL/Data/Kidney\\ \\(3T,\\ Philips,\\ UCH,\\ 20150713\\)/Nifti/"
    filenames = [
        "20150713_09583130x3mmlongSENSEs2801a1028",
        "20150713_09583130x3mmlongSENSEs2701a1027",
        "20150713_09583130x3mmlongSENSEs2601a1026"
        ]


    N = len(filenames)
    target_stack_id = 0


    clear_dir_out = False

    if clear_dir_out:
        cmd = "rm -rf " + dir_out + "*"
        os.system(cmd)


    ## Data Preprocessing:
    data_preprocessing = DataPreprocessing(dir_out, dir_input, filenames)   
    # data_preprocessing.normalize_images()

    # data_preprocessing.create_rectangular_mask(dir_input, filenames[1], [161, 219], 100, 100)
    # data_preprocessing.create_rectangular_mask(dir_input, filenames[2], [150, 214], 120, 150)

    ## Make images "ready": Progagate segmentations + subsequent mask to target volume
    data_preprocessing.segmentation_propagation(target_stack_id)
    
    # data_preprocessing.crop_and_copy_images()


    ## Use already "ready" images for further processing:
    # data_preprocessing.copy_files(dir_out+"input_data/", dir_input, filenames)
    

    stacks = data_preprocessing.get_stacks()


    # ## 
    # registration = Registration(stacks, target_stack_id)
    # registration.register_images()
    # registration.compute_HR_volume()

    # target = registration.get_target_stack()
    # corrected_stacks = registration.get_corrected_stacks()

    # for i in range(0,len(corrected_stacks)):
    #     print corrected_stacks[i].get_filenames()

   

    # display_image(registration.get_target_stack().get_data()[:,:,10])

    # plt.imshow(registration.get_target_stack().get_data()[:,:,10], cmap="Greys_r")

    # img = registration.get_HR_volume().get_data()
    # N = img.shape

    # slice_0 = img[:,:,np.round(N[2]/2)]
    # slice_1 = img[:,np.round(N[1]/2),:]
    # slice_2 = img[np.round(N[0]/2),:,:]

    # show_slices([slice_0,slice_1,slice_2])
    # plt.show()



    """ #######################################################################
        Playground:
    """
    flag_playground = 0

    if flag_playground:

        ind = 7

        nifti_0 = nib.load(dir_out+filenames[target_stack_id]+".nii.gz")
        nifti_1 = nib.load(dir_out+filenames[ind]+".nii.gz")

        nifti_0_warp = nib.load(dir_out+filenames[target_stack_id]+".nii.gz")
        nifti_1_warp = nib.load(dir_out+filenames[ind]+".nii.gz")

        T_0 = np.loadtxt(dir_out+filenames[target_stack_id]+".txt")
        T_1 = np.loadtxt(dir_out+filenames[ind]+".txt")

        M_0 = nifti_0.affine
        M_1 = nifti_1.affine

        M_0_warp = nifti_0_warp.affine
        M_1_warp = nifti_1_warp.affine

        print M_0
        print np.dot(M_1[0:-1,1],M_1[0:-1,2])

        print M_0-M_0_warp
        # print T_0-npl.inv(M_0_warp)
        print M_0-M_1_warp
        # print npl.inv(T_1)*M_0-M_1_warp
        # print npl.inv(T_1)*M_0-M_1_warp

        npl.inv(T_1*M_1)*M_0


        # print nib.aff2axcodes(nifti.affine)
        # print nib.zooms

        # array =  stacks[0].data
        # FS = stacks[0]
        # FS_copy = copy.copy(stacks[0])
        # FS_deep = copy.deepcopy(stacks[0])

        # stacks[0].set_data(array*0)




if __name__ == "__main__":
    main()
