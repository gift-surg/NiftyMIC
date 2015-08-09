#!/usr/bin/python

## Import libraries 
import os                       # used to execute terminal commands in python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import nibabel as nib           # nifti files
import unittest

import numpy as np
import numpy.linalg as npl
import sys

## Import other py-files within src-folder
sys.path.insert(0, '../src')
from DataPreprocessing import *
from SliceStack import *
from FileAndImageHelpers import *
from Registration import *
from MyImageProcessingFunctions import *
from TwoDImage import *
# from SegmentationPropagation import *
# from SimilarityMeasures import *
# from StatisticalAnalysis import *

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
    "20150713_09583130x3mmlongSENSEs2601a1026",
    "20150713_09583130x3mmlongSENSEs2501a1025",
    "20150713_09583130x3mmlongSENSEs2401a1024",
    "20150713_09583130x3mmlongSENSEs2301a1023"
    ]


## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestUM(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_translation_of_single_slice_1(self):
        data_preprocessing = DataPreprocessing(dir_out, dir_input, filenames[0:3])   
        img = SliceStack(dir_out+"input_data/","0")
        single_slices = img.get_single_slices()
        v0 = np.dot(single_slices[0].get_affine(), np.array([0,0,0,1]))
        v1 = np.dot(single_slices[1].get_affine(), np.array([0,0,0,1]))
        v2 = np.dot(single_slices[2].get_affine(), np.array([0,0,0,1]))
        self.assertEqual( np.around(np.sum(abs(v2-v1-(v1-v0))), decimals=4), 0 )
 

    ## Test based on difference to voxel size
    def test_translation_of_single_slice_2(self):
        img = SliceStack(dir_out+"input_data/","0")
        pixdim = img.get_header().get_zooms()    # voxel sizes in mm

        single_slices = img.get_single_slices()
        v0 = np.dot(single_slices[0].get_affine(), np.array([0,0,0,1]))
        v1 = np.dot(single_slices[1].get_affine(), np.array([0,0,0,1]))
        self.assertEqual( np.around(abs(np.linalg.norm(v1-v0)-pixdim[2]), decimals=4), 0 )
 
if __name__ == '__main__':
    unittest.main()