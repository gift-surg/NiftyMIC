## \file MyImageProcessingHelpers.py
#  \brief  
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date January 2016


## Import libraries
import SimpleITK as sitk
import itk

import numpy as np
import unittest

import os                       # used to execute terminal commands in python

## Change viewer for sitk.Show command
#%env SITK_SHOW_COMMAND /Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP

"""
Functions
"""
## Show image with ITK-Snap. Image is saved to /tmp/ for that purpose
#  \param[in] image_itk image to show
#  \param[in] segmentation_itk 
#  \param[in] overlay_itk image which shall be overlayed onto image_itk (optional)
#  \param[in] title filename for file written to /tmp/ (optional)
def show_itk_image(image_itk, segmentation_itk=None, overlay_itk=None, title="test"):
    
    dir_output = "/tmp/"
    # cmd = "fslview " + dir_output + title + ".nii.gz & "

    ## Define type of output image
    pixel_type = itk.D
    dim = image_itk.GetBufferedRegion().GetImageDimension()
    image_type = itk.Image[pixel_type, dim]

    ## Define writer
    writer = itk.ImageFileWriter[image_type].New()
    # writer_2D.Update()
    # image_IO_type = itk.NiftiImageIO

    ## Write image_itk
    writer.SetInput(image_itk)
    writer.SetFileName(dir_output + title + ".nii.gz")
    writer.Update()

    if overlay_itk is not None and segmentation_itk is None:
        ## Write overlay_itk:
        writer.SetInput(overlay_itk)
        writer.SetFileName(dir_output + title + "_overlay.nii.gz")
        writer.Update()

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-o " + dir_output + title + "_overlay.nii.gz " \
            "& "
    
    elif overlay_itk is None and segmentation_itk is not None:
        ## Write segmentation_itk:
        writer.SetInput(segmentation_itk)
        writer.SetFileName(dir_output + title + "_segmentation.nii.gz")
        writer.Update()

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-s " + dir_output + title + "_segmentation.nii.gz " \
            + "& "

    elif overlay_itk is not None and segmentation_itk is not None:
        ## Write overlay_itk:
        writer.SetInput(overlay_itk)
        writer.SetFileName(dir_output + title + "_overlay.nii.gz")
        writer.Update()

        ## Write segmentation_itk:
        writer.SetInput(segmentation_itk)
        writer.SetFileName(dir_output + title + "_segmentation.nii.gz")
        writer.Update()

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-s " + dir_output + title + "_segmentation.nii.gz " \
            + "-o " + dir_output + title + "_overlay.nii.gz " \
            + "& "

    else:
        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            "& "

    os.system(cmd)

    return None

## Show image with ITK-Snap. Image is saved to /tmp/ for that purpose
#  \param[in] image_sitk image to show
#  \param[in] segmentation_sitk 
#  \param[in] overlay_sitk image which shall be overlayed onto image_sitk (optional)
#  \param[in] title filename for file written to /tmp/ (optional)
def show_sitk_image(image_sitk, segmentation_sitk=None, overlay_sitk=None, title="test"):
    
    dir_output = "/tmp/"
    # cmd = "fslview " + dir_output + title + ".nii.gz & "

    if overlay_sitk is not None and segmentation_sitk is None:
        sitk.WriteImage(image_sitk, dir_output + title + ".nii.gz")
        sitk.WriteImage(overlay_sitk, dir_output + title + "_overlay.nii.gz")

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-o " + dir_output + title + "_overlay.nii.gz " \
            + "& "

    elif overlay_sitk is None and segmentation_sitk is not None:
        sitk.WriteImage(image_sitk, dir_output + title + ".nii.gz")
        sitk.WriteImage(segmentation_sitk, dir_output + title + "_segmentation.nii.gz")

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-s " + dir_output + title + "_segmentation.nii.gz " \
            + "& "

    elif overlay_sitk is not None and segmentation_sitk is not None:
        sitk.WriteImage(image_sitk, dir_output + title + ".nii.gz")
        sitk.WriteImage(segmentation_sitk, dir_output + title + "_segmentation.nii.gz")
        sitk.WriteImage(overlay_sitk, dir_output + title + "_overlay.nii.gz")

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-s " + dir_output + title + "_segmentation.nii.gz " \
            + "-o " + dir_output + title + "_overlay.nii.gz " \
            + "& "

    else:
        sitk.WriteImage(image_sitk, dir_output + title + ".nii.gz")

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "& "

    os.system(cmd)

    return None


## Create an image which only contains one centered dot with max intensity
#  Works for 2D and 3D images
#  \param[in] image_sitk sitk::Image serving which gets cleared out
#  \param[in] filename filename for the generated single-dot image (optional)
#  \return single-dot image is written to the specified location if given
def create_image_with_single_dot(image_sitk, filename=None):
    nda = sitk.GetArrayFromImage(image_sitk)
    nda[:] = 0
    shape = np.array(nda.shape)

    ## This choice of midpoint will center the max-intensity voxel with the
    #  center-point the viewers ITK-Snap and FSLView use.
    midpoint = tuple((shape/2.).astype(int))

    nda[midpoint]=100 #could also be seen as percent!

    image_SingleDot = sitk.GetImageFromArray(nda)
    image_SingleDot.CopyInformation(image_sitk)

    show_sitk_image(image_SingleDot, title="SingleDot_" + str(len(shape)) + "D")

    if filename is not None:
        sitk.WriteImage(image_SingleDot, filename)


"""
Unit Test Class
"""
class TestUM(unittest.TestCase):

    accuracy = 10
    dir_input = "data/"
    dir_output = "results/"


    def setUp(self):
        pass

"""
Main Function
"""
if __name__ == '__main__':
    dir_input = "data/"
