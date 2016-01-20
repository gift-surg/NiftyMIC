#!/usr/bin/python

## \file ITK_GaussianInterpolator.py
#  \brief Figure out how to edit itkGaussianInterpolateImageFunction.hxx
#
#  \author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date: January 2016

## Import libraries
import SimpleITK as sitk
import itk
# import nibabel as nib
import numpy as np
import unittest
import math                     # for error function (erf)

import os                       # used to execute terminal commands in python
import sys
sys.path.append("../src/")

## Import modules from src-folder
import SimpleITKHelper as sitkh

"""
Functions
"""

"""
Unit Test Class
"""

class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_01(self):
        pass


"""
Main Function
"""
if __name__ == '__main__':

    class Object(object):
        pass

    dir_input = "data/"
    dir_output = "results/"
    filename = "fetal_brain_a"
    # filename = "CTL_0_baseline_deleted_0.5"

    ## Define types of input and output pixels and state dimension of images
    input_pixel_type = itk.F
    output_pixel_type = input_pixel_type

    input_dimension = 3
    output_dimension = input_dimension

    ## Define type of input and output image
    input_image_type = itk.Image[input_pixel_type, input_dimension]
    output_image_type = itk.Image[output_pixel_type, output_dimension]

    ## Instantiate types of reader and writer
    reader_type = itk.ImageFileReader[input_image_type]
    writer_type = itk.ImageFileWriter[output_image_type]
    image_IO_type = itk.NiftiImageIO

    ## Create reader and writer
    reader = reader_type.New()
    writer = writer_type.New()

    ## Set image IO type to nifti
    image_IO = image_IO_type.New()
    reader.SetImageIO(image_IO)

    ## Read image
    reader.SetFileName(dir_input + filename + ".nii.gz")
    reader.Update()

    ## Get image
    image_itk = reader.GetOutput()

    image_sitk = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat32)

    """
    Start for itkGaussianInterpolateImageFunction:
    """
    sigma = 2
    Sigma = np.array([sigma,sigma,sigma])
    alpha = 2
    cindex = np.array([10, 28, 0])

    size = image_itk.GetBufferedRegion().GetSize()
    spacing = image_itk.GetSpacing()
    ImageDimension = len(size)

    BoundingBoxStart = np.zeros(ImageDimension)
    BoundingBoxEnd = np.zeros(ImageDimension)
    ScalingFactor = np.zeros(ImageDimension)
    CutoffDistance = np.zeros(ImageDimension)

    boundingBoxSize = np.zeros(ImageDimension).astype(int)
    begin = np.zeros(ImageDimension).astype(int)
    end = np.zeros(ImageDimension).astype(int)

    region = Object()
    region.index = np.zeros(ImageDimension).astype(int)
    region.size = np.zeros(ImageDimension).astype(int)

    erf = Object()

    for d in range(0, ImageDimension):
        BoundingBoxStart[d] = -0.5
        BoundingBoxEnd[d] = size[d] - 0.5
        ScalingFactor[d] = 1/(np.sqrt(2)*Sigma[d]/spacing[d])
        CutoffDistance[d] = Sigma[d]*alpha/spacing[d]
        
    # CutoffDistance[:] = 2.2

    print("size = %s" %(size))
    print("spacing = %s" %(spacing))
    print("cindex = %s" %(cindex))

    print("BoundingBoxStart = %s" %(BoundingBoxStart))
    print("BoundingBoxEnd = %s" %(BoundingBoxEnd))
    print("ScalingFactor = %s" %(ScalingFactor))
    print("CutoffDistance = %s" %(CutoffDistance))


    for d in range(0, ImageDimension):
        boundingBoxSize[d] = (BoundingBoxEnd[d] - BoundingBoxStart[d] + 0.5).astype(int) # = size[d]
        begin[d] = np.max([0, (np.floor(cindex[d] - BoundingBoxStart[d] - CutoffDistance[d])).astype(int)])
        end[d] = np.min([boundingBoxSize[d], (np.ceil(cindex[d] - BoundingBoxStart[d] + CutoffDistance[d])).astype(int)])
        region.index[d] = begin[d]
        region.size[d] = end[d] - begin[d]

        print("\nd=%s:" %d)
        print("boundingBoxSize[d] = %s" %(boundingBoxSize[d]))
        print("begin[d] = %s" %(begin[d]))
        print("end[d] = %s" %(end[d]))
        print("region.index[d] = %s" %(region.index[d]))
        print("region.size[d] = %s" %(region.size[d]))


    ## ComputeErrorFunctionArray
    erfArray = [None]*ImageDimension

    for d in range(0, ImageDimension):     
        ## Create erfArray
        erfArray[d] = np.zeros(boundingBoxSize[d])

        ## Start at first voxel of image
        t = (BoundingBoxStart[d] - cindex[d] + begin[d])*ScalingFactor[d]
        e_last = math.erf(t)

        for i in range(begin[d], end[d]):
            t += ScalingFactor[d]
            e_now = math.erf(t)
            erfArray[d][i] = e_now - e_last
            e_last = e_now


    ## 
    for i in range(0, ):
        

    # print erfArray[0]
    # print erfArray[1]
    # print erfArray[2]


    """
    Unit tests:
    """
    # print("\nUnit tests:\n--------------")
    # unittest.main()