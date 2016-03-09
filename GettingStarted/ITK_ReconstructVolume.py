## \file ITK_ReconstructVolume.py
#  \brief  Translate algorithms which were tested in Optimization.py into
#       something which performs volume reconstructions from slices
#       given the ITK/SimpleITK framework
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date March 2016

import SimpleITK as sitk
import itk

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import leastsq
from scipy import ndimage
import unittest
import matplotlib.pyplot as plt

import sys
import time
sys.path.append("../src")

import SimpleITKHelper as sitkh
import Stack as st
import Slice as sl
import ReconstructionManager as rm


"""
Classes
"""

## Define type of pixel and image in ITK
pixel_type = itk.D
image_type = itk.Image[pixel_type, 3]

class Optimize:

    def __init__(self, stacks, HR_volume):

            ## Initialize variables
            self._stacks = stacks
            self._HR_volume = HR_volume
            self._N_stacks = len(stacks)

            self._alpha_cut = 3

            ## Append itk objects
            self._append_itk_object_on_slices_of_stacks()
            self._HR_volume.itk = sitkh.convert_sitk_to_itk_image(self._HR_volume.sitk)


            ## Allocate and initialize Oriented Gaussian Interpolate Image Filter
            self._filter_oriented_Gaussian_interpolator = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
            self._filter_oriented_Gaussian_interpolator.SetAlpha(self._alpha_cut)
            
            self._filter_oriented_Gaussian = itk.ResampleImageFilter[image_type, image_type].New()
            self._filter_oriented_Gaussian.SetInterpolator(self._filter_oriented_Gaussian_interpolator)
            self._filter_oriented_Gaussian.SetDefaultPixelValue( 0.0 )

            ## Allocate and initialize Adjoint Oriented Gaussian Interpolate Image Filter
            self._filter_adjoint_oriented_Gaussian = itk.AdjointOrientedGaussianInterpolateImageFilter[image_type, image_type].New()
            self._filter_adjoint_oriented_Gaussian.SetDefaultPixelValue( 0.0 )
            self._filter_adjoint_oriented_Gaussian.SetAlpha(self._alpha_cut)
            self._filter_adjoint_oriented_Gaussian.SetOutputParametersFromImage( self._HR_volume.itk )


    ##
    #  \return current estimate of HR volume, instance of sitk.Image
    def get_HR_volume_sitk(self):
        return self._HR_volume_sitk


    def set_alpha(self, alpha_cut):
        self._alpha_cut = alpha_cut

        ## Update cut-off distance for both image filters
        self._filter_oriented_Gaussian_interpolator.SetAlpha(alpha_cut)
        self._filter_adjoint_oriented_Gaussian.SetAlpha(alpha_cut)


    def get_alpha(self):
        return self._alpha_cut


    ## Run reconstruction algorithm
    def run_reconstruction(self):

        duplicator = itk.ImageDuplicator[image_type].New()
        duplicator.SetInputImage(self._HR_volume.itk)
        duplicator.Update()

        HR_volume = duplicator.GetOutput()
        HR_volume.DisconnectPipeline()                

        tmp = self._sum_ATA(HR_volume)

        sitkh.show_itk_image(tmp, overlay_itk=self._HR_volume.itk, title="sum_ATA+HR")

        # LR = self._A(self._HR_volume.itk)
        # HR = self._AT(slice.itk)

        # sitkh.show_itk_image(slice.itk,overlay_itk=LR,title="LR")
        # sitkh.show_itk_image(self._HR_volume.itk,overlay_itk=HR,title="HR")


    ## Append slices as itk image on each object Slice
    #  HACK!
    def _append_itk_object_on_slices_of_stacks(self):

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            for j in range(0, stack.get_number_of_slices()):
                slice = slices[j]

                slice.itk = sitkh.convert_sitk_to_itk_image(slice.sitk)
                

    ## Compute the covariance matrix modelling the PSF in-plane and 
    #  through-plane of a slice.
    #  The PSF is modelled as Gaussian with
    #       FWHM = 1.2*in-plane-resolution (in-plane)
    #       FWHM = slice thickness (through-plane)
    #  \param[in] slice Slice instance defining the PSF
    #  \return Covariance matrix representing the PSF modelled as Gaussian
    def _get_PSF_covariance_matrix(self, slice):
        spacing = np.array(slice.sitk.GetSpacing())

        ## Compute Gaussian to approximate in-plane PSF:
        sigma_x2 = (1.2*spacing[0])**2/(8*np.log(2))
        sigma_y2 = (1.2*spacing[1])**2/(8*np.log(2))

        ## Compute Gaussian to approximate through-plane PSF:
        sigma_z2 = spacing[2]**2/(8*np.log(2))

        return np.diag([sigma_x2, sigma_y2, sigma_z2])


    ## Compute rotated covariance matrix which expresses the PSF of the slice
    #  in the coordinates of the HR_volume
    #  \param[in] slice slice which is aimed to be simulated according to the slice acquisition model
    #  \return Covariance matrix U*Sigma_diag*U' where U represents the
    #          orthogonal trafo between slice and HR_volume
    def _get_PSF_covariance_matrix_HR_volume_coordinates(self, slice):

        ## Compute rotation matrix to express the PSF in the coordinate system of the HR volume
        dim = slice.sitk.GetDimension()
        direction_matrix_HR_volume = np.array(self._HR_volume.sitk.GetDirection()).reshape(dim,dim)
        direction_matrix_slice = np.array(slice.sitk.GetDirection()).reshape(dim,dim)

        U = direction_matrix_HR_volume.transpose().dot(direction_matrix_slice)
        # print("U = \n%s\ndet(U) = %s" % (U,np.linalg.det(U)))

        ## Get axis algined PSF
        cov = self._get_PSF_covariance_matrix(slice)

        ## Return Gaussian blurring variance covariance matrix of slice in HR volume coordinates 
        return U.dot(cov).dot(U.transpose())


    ## Update internal Oriented and Adjoint Oriented Gaussian Interpolate Image Filter parameters 
    #  according to the relative position between slice and HR volume
    #  \param[in] slice Slice object
    def _update_oriented_adjoint_oriented_Gaussian_image_filters(self, slice):
        ## Get variance covariance matrix representing Gaussian blurring in HR volume coordinates
        Cov_HR_coord = self._get_PSF_covariance_matrix_HR_volume_coordinates( slice )

        ## Update parameters of forward operator A
        self._filter_oriented_Gaussian_interpolator.SetCovariance( Cov_HR_coord.flatten() )
        self._filter_oriented_Gaussian.SetOutputParametersFromImage( slice.itk )
        
        ## Update parameters of backward/adjoint operator A'
        self._filter_adjoint_oriented_Gaussian.SetCovariance( Cov_HR_coord.flatten() )

        #  (Slice does not contain ITK object so either conversion to ITK object or converting by hand)
        #  (Conversion by hand does not allow to set OutputStartIndex)
        # self._filter_oriented_Gaussian.SetOutputOrigin( slice.sitk.GetOrigin() )
        # self._filter_oriented_Gaussian.SetOutputSpacing( slice.sitk.GetSpacing() )
        # self._filter_oriented_Gaussian.SetOutputDirection( sitkh.get_itk_direction_from_sitk_image(slice.sitk) )
        # # self._filter_oriented_Gaussian.SetOutputStartIndex( "Does not exist in sitk" )
        # self._filter_oriented_Gaussian.SetSize( slice.sitk.GetSize() )
        

    ## Perform forward operation on HR image, i.e. y = DB(x) with D and B being 
    #  the downsampling and blurring operator, respectively.
    #  \param[in] HR_volume_itk HR image as itk.Image object
    #  \return image in LR space as itk.Image object after performed forward operation
    def _A(self, HR_volume_itk):
        HR_volume_itk.Update()
        self._filter_oriented_Gaussian.SetInput( HR_volume_itk )
        self._filter_oriented_Gaussian.UpdateLargestPossibleRegion()
        self._filter_oriented_Gaussian.Update()

        slice_itk = self._filter_oriented_Gaussian.GetOutput();
        slice_itk.DisconnectPipeline()

        return slice_itk


    ## Perform backward operation on LR image, i.e. z = B'D'(y) with D' and B' being 
    #  the adjoint downsampling and blurring operator, respectively.
    #  \param[in] slice_itk LR image as itk.Image object
    #  \return image in HR space as itk.Image object after performed backward operation
    def _AT(self, slice_itk):
        self._filter_adjoint_oriented_Gaussian.SetInput( slice_itk )
        self._filter_adjoint_oriented_Gaussian.Update()

        HR_volume_itk = self._filter_adjoint_oriented_Gaussian.GetOutput()
        HR_volume_itk.DisconnectPipeline()

        return HR_volume_itk


    ## Perform the operation A'A(x) with A = DB, i.e. the combination of
    #  downsampling D and blurring B.
    #  \param[in] HR_volume_itk HR image as itk.Image object
    #  \return image in HR space as itk.Image object after performed forward 
    #          and backward operation
    def _ATA(self, HR_volume_itk):
        return self._AT( self._A(HR_volume_itk) )


    ## Compute sum_{k=1}^K A_k' A_k x
    #  \param[in] HR_volume_itk HR image as itk.Image object
    #  \return sum of all forward an back projected operations of image as
    #       itk.Image object
    def _sum_ATA(self, HR_volume_itk):

        ## Create image adder
        adder = itk.AddImageFilter[image_type, image_type, image_type].New()
        ## This will copy the buffer of input1 to the output and remove the need to allocate a new output
        adder.InPlaceOn() 

        ## Duplicate HR_volume with zero image buffer
        duplicator = itk.ImageDuplicator[image_type].New()
        duplicator.SetInputImage(HR_volume_itk)
        duplicator.Update()

        sum_ATAx = duplicator.GetOutput()
        sum_ATAx.DisconnectPipeline()
        sum_ATAx.FillBuffer(0.0)

        ## Insert zero image for image adder
        adder.SetInput1(sum_ATAx)


        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            for j in range(0, stack.get_number_of_slices()):

                slice = slices[j]
                # sitkh.show_itk_image(slice.itk,title="slice")

                ## Update A_k and A_k' based on position of slice
                self._update_oriented_adjoint_oriented_Gaussian_image_filters(slice)

                ## Perform A_k'A_k(x) 
                ATA_k_x = self._ATA(HR_volume_itk)

                ## Add contribution
                adder.SetInput2(ATA_k_x)
                adder.Update()

                sum_ATAx = adder.GetOutput()
                sum_ATAx.DisconnectPipeline()

                ## Prepare for next cycle
                adder.SetInput1(sum_ATAx)

        return sum_ATAx




"""
Main Function
"""
if __name__ == '__main__':
    
    PIG = False
    # PIG = True
    
    if PIG:
        ## Data of structural pig
        dir_input = "../data/StructuralData_Pig/"
        filenames = [
            "T22D3mm05x05hresCLEARs601a1006",
            "T22D3mm05x05hresCLEARs701a1007",
            "T22D3mm05x05hresCLEARs901a1009"
            ]
        filename_HR_volume = "3DBrainViewT2SHCCLEARs1301a1013"

    ## Data of GettingStarted folder
    else:
        dir_input = "data/"
        filenames = [
            "FetalBrain_stack0_registered",
            "FetalBrain_stack1_registered",
            "FetalBrain_stack2_registered"
            ]
        filename_HR_volume = "FetalBrain_reconstruction_4stacks"

    ## Output folder
    dir_output = "results/"

    ## Prepare output directory
    reconstruction_manager = rm.ReconstructionManager(dir_output)

    ## Read input data
    reconstruction_manager.read_input_data(dir_input, filenames)

    HR_volume = st.Stack.from_nifti(dir_input,filename_HR_volume)
    stacks = reconstruction_manager.get_stacks()
    N_stacks = len(stacks)

    stack = stacks[1]
    slices = stack.get_slices()

    MyOptimizer = Optimize(stacks, HR_volume)

    # sitkh.show_sitk_image(HR_volumesitk)


    # stacks[1].get_slice(0).write(directory=dir_output, filename="slice")
    # HR_volume.write(directory=dir_output, filename="HR_volume")


    MyOptimizer.run_reconstruction()


