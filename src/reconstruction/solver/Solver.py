#!/usr/bin/python

## \file Solver.py
#  \brief Implementation of basis class to solve the slice acquisition model
#  \f[ 
#       y_k = D_k B_k W_k x = A_k x 
#  \f]
#  for each slice \f$ y_k,\,k=1,\dots,K \f$ during reconstruction
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date July 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np

## Add directories to import modules
dir_src_root = "../src/"
sys.path.append( dir_src_root + "base/" )

## Import modules
import SimpleITKHelper as sitkh
import PSF as psf

## Pixel type of used 3D ITK image
PIXEL_TYPE = itk.D

## ITK image type 
IMAGE_TYPE = itk.Image[PIXEL_TYPE, 3]


class Solver():

    ## Constructor
    #  \param[in] stacks list of Stack objects containing all stacks used for the reconstruction
    #  \param[in] HR_volume Stack object containing the current estimate of the HR volume (used as initial value + space definition)
    #  \param[in] alpha_cut Cut-off distance for Gaussian blurring filter
    def __init__(self, stacks, HR_volume, alpha_cut=3):

        ## Initialize variables
        self._stacks = stacks
        self._HR_volume = HR_volume
        self._N_stacks = len(stacks)

        ## Used for PSF modelling
        self._psf = psf.PSF()

        ## Cut-off distance for Gaussian blurring filter
        self._alpha_cut = alpha_cut  

        ## Allocate and initialize Oriented Gaussian Interpolate Image Filter
        self._filter_oriented_Gaussian_interpolator = itk.OrientedGaussianInterpolateImageFunction[IMAGE_TYPE, PIXEL_TYPE].New()
        self._filter_oriented_Gaussian_interpolator.SetAlpha(self._alpha_cut)
        
        self._filter_oriented_Gaussian = itk.ResampleImageFilter[IMAGE_TYPE, IMAGE_TYPE].New()
        self._filter_oriented_Gaussian.SetInterpolator( self._filter_oriented_Gaussian_interpolator )
        self._filter_oriented_Gaussian.SetDefaultPixelValue( 0.0 )

        ## Allocate and initialize Adjoint Oriented Gaussian Interpolate Image Filter
        self._filter_adjoint_oriented_Gaussian = itk.AdjointOrientedGaussianInterpolateImageFilter[IMAGE_TYPE, IMAGE_TYPE].New()
        self._filter_adjoint_oriented_Gaussian.SetDefaultPixelValue( 0.0 )
        self._filter_adjoint_oriented_Gaussian.SetAlpha(self._alpha_cut)
        self._filter_adjoint_oriented_Gaussian.SetOutputParametersFromImage( self._HR_volume.itk )

        ## Create PyBuffer object for conversion between NumPy arrays and ITK images
        self._itk2np = itk.PyBuffer[IMAGE_TYPE]

        ## Extract information ready to use for itk image conversion operations
        self._HR_shape_nda = sitk.GetArrayFromImage( self._HR_volume.sitk ).shape


    ## Get current estimate of HR volume
    #  \return current estimate of HR volume, instance of Stack
    def get_HR_volume(self):
        return self._HR_volume


    ## Set cut-off distance
    #  \param[in] alpha_cut scalar value
    def set_alpha_cut(self, alpha_cut):
        self._alpha_cut = alpha_cut


    ## Get cut-off distance
    #  \return scalar value
    def get_alpha_cut(self):
        return self._alpha_cut
        

    ## Perform forward operation on HR image, i.e. \f$y = DBx =: Ax \f$ with \f$D\f$  and \f$ B \f$ being 
    #  the downsampling and blurring operator, respectively.
    #  \param[in] HR_volume_itk HR image as itk.Image object
    #  \param[in] slice_k Slice object which defines operator A_k
    #  \return image in LR space as itk.Image object after performed forward operation
    def Ak(self, HR_volume_itk, slice_k):

        ## Set up operator A_k based on relative position to HR volume and their dimensions
        self._update_oriented_adjoint_oriented_Gaussian_image_filters(slice_k)

        ## Perform forward operation A_k on HR volume object
        HR_volume_itk.Update()
        self._filter_oriented_Gaussian.SetInput( HR_volume_itk )
        self._filter_oriented_Gaussian.UpdateLargestPossibleRegion()
        self._filter_oriented_Gaussian.Update()

        slice_itk = self._filter_oriented_Gaussian.GetOutput();
        slice_itk.DisconnectPipeline()

        return slice_itk


    ## Perform backward operation on LR image, i.e. \f$z = B^*D^*My = A^*My \f$ with \f$ D^* \f$ and \f$ B^* \f$ being 
    #  the adjoint downsampling and blurring operator, respectively.
    #  \param[in] slice_itk LR image as itk.Image object
    #  \param[in] slice_k Slice object which defines operator A_k^*
    #  \return image in HR space as itk.Image object after performed backward operation
    def Ak_adj(self, slice_itk, slice_k):

        ## Set up operator A_k^* based on relative position to HR volume and their dimensions
        self._update_oriented_adjoint_oriented_Gaussian_image_filters(slice_k)

        ## Perform backward operation A_k^* on LR image object
        self._filter_adjoint_oriented_Gaussian.SetInput( slice_itk )
        self._filter_adjoint_oriented_Gaussian.UpdateLargestPossibleRegion()
        self._filter_adjoint_oriented_Gaussian.Update()

        HR_volume_itk = self._filter_adjoint_oriented_Gaussian.GetOutput()
        HR_volume_itk.DisconnectPipeline()

        return HR_volume_itk


    ## Evaluate \f$ D \vec{x} \f$ 
    #  \f$ = \begin{pmatrix} D_x \\ D_y \\ D_z \end{pmatrix} \vec{x}\f$ 
    #  within the adjoint augmented linear operator for TK1-regularization.
    #  \param[in] HR_nda_vec HR data as 1D array
    #  \return evaluated differential operator as part of
    #       augmented linear operator as 1D array
    def D(self, HR_nda_vec):
        
        ## Number of voxels always given by 3 times HR voxels
        N_voxels = self._N_voxels_HR_volume

        ## Allocate memory
        D_x = np.zeros(3*N_voxels)

        HR_nda = HR_nda_vec.reshape( self._HR_shape_nda )

        ## Differentiate w.r.t. x, y and z and fill corresponding indices
        D_x[0:N_voxels] = self._differential_operations.Dx(HR_nda).flatten()
        D_x[N_voxels:2*N_voxels] = self._differential_operations.Dy(HR_nda).flatten()
        D_x[2*N_voxels:3*N_voxels] = self._differential_operations.Dz(HR_nda).flatten()

        return D_x


    ## Evaluate \f$ D^* \vec{y} \f$ 
    #  \f$ = \begin{pmatrix} D_x^* && D_y && D_z \end{pmatrix} \vec{y}\in\mathbb{R}^N\f$ 
    #  within the adjoint augmented linear operator for TK1-regularization.
    #  \param[in] stacked_slices_nda_vec stacked slice data as 1D array
    #  \return evaluated adjoint differential operator as part of 
    #       augmented adjoint linear operator as 1D array
    def D_adj(self, stacked_slices_nda_vec):

        ## Number of voxels always given by 3 times HR voxels
        N_voxels = self._N_voxels_HR_volume
        
        ## Allocate memory
        D_adj_slice = np.zeros(3*N_voxels)

        ## Extract respective x, y and z groups within compound stacked_slices_nda_vec
        slice_x_nda_vec = stacked_slices_nda_vec[-3*N_voxels: -2*N_voxels]
        slice_y_nda_vec = stacked_slices_nda_vec[-2*N_voxels: -N_voxels]
        slice_z_nda_vec = stacked_slices_nda_vec[-N_voxels:]

        ## Reshape in order to apply differentiation
        slice_x_nda = slice_x_nda_vec.reshape( self._HR_shape_nda )
        slice_y_nda = slice_y_nda_vec.reshape( self._HR_shape_nda )
        slice_z_nda = slice_z_nda_vec.reshape( self._HR_shape_nda )

        ## Apply adjoint differentiation w.r.t. x, y and z
        Dx_adj_vec = self._differential_operations.Dx_adj(slice_x_nda).flatten()
        Dy_adj_vec = self._differential_operations.Dy_adj(slice_y_nda).flatten()
        Dz_adj_vec = self._differential_operations.Dz_adj(slice_z_nda).flatten()

        ## Return added contributions 
        return  Dx_adj_vec + Dy_adj_vec + Dz_adj_vec

    ## Masking operation M_k
    #  \param[in] slice_itk image in LR space as itk.Image object
    #  \param[in] slice_k Slice object which defines operator M_k
    def Mk(self, slice_itk, slice_k):

        ## Perform masking M_k based
        multiplier = itk.MultiplyImageFilter[IMAGE_TYPE, IMAGE_TYPE, IMAGE_TYPE].New()
        multiplier.SetInput1( slice_k.itk_mask )
        multiplier.SetInput2( slice_itk )
        multiplier.Update()

        Mk_slice_itk = multiplier.GetOutput()
        Mk_slice_itk.DisconnectPipeline()

        return Mk_slice_itk


    ## Operation M_k A_k x
    #  \param[in] HR_volume_itk HR image as itk.Image object
    #  \param[in] slice_k Slice object which defines operator M_k and A_k
    def Mk_Ak(self, HR_volume_itk, slice_k):

        ## Compute A_k x
        Ak_HR_volume_itk = self.Ak(HR_volume_itk, slice_k)

        ## Compute M_k A_k x
        return self.Mk(Ak_HR_volume_itk, slice_k)


    ## Operation A_k^* M_k y_k
    #  \param[in] slice_itk LR image as itk.Image object
    #  \param[in] slice_k Slice object which defines operator A_k^*
    #  \return image in HR space as itk.Image object after performed backward operation
    def Ak_adj_Mk(self, slice_itk, slice_k):

        ## Compute M_k y_k
        Mk_slice_itk = self.Mk(slice_itk, slice_k)

        ## Compute A_k^* M_k y_k
        return self.Ak_adj(Mk_slice_itk, slice_k)


    ## Evaluate
    #  \f$ MA \vec{x} \f$
    #  \f$ = \begin{pmatrix} M_1 A_1 \\ M_2 A_2 \\ \vdots \\ M_K A_K \end{pmatrix} \vec{x} \f$
    #  \param[in] HR_nda_vec HR data as 1D array
    #  \return evaluated MAx as part of augmented linear operator as 1D array
    def MA(self, HR_nda_vec):

        ## Convert HR data array back to itk.Image object
        x_itk = self.get_itk_image_from_array_vec(HR_nda_vec, self._HR_volume.itk)
        
        ## Allocate memory
        MA_x = np.zeros(self._N_total_slice_voxels)

        ## Define index for first voxel of first slice within array
        i_min = 0

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            ## Get number of voxels of each slice in current stack
            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for j in range(0, stack.get_number_of_slices()):
                
                ## Define index for last voxel to specify current slice (exlusive)
                i_max = i_min + N_slice_voxels

                ## Compute M_k A_k y_k
                slice_itk = self.Mk_Ak(x_itk, slices[j])
                slice_nda = self._itk2np.GetArrayFromImage(slice_itk)

                ## Fill corresponding elements
                MA_x[i_min:i_max] = slice_nda.flatten()

                ## Define index for first voxel to specify subsequent slice (inclusive)
                i_min = i_max

        return MA_x


    ## Evaluate
    #  \f$ A^* M \vec{y}
    #     = \begin{bmatrix} A_1^* M_1 && A_2^* M_2 && \cdots && A_K^* M_K \end{bmatrix} \vec{y}
    #  \f$
    #  \param[in] stacked_slices_nda_vec stacked slice data as 1D array
    #  \return evaluated A'My as part of augmented adjoint linear operator as 1D array
    def A_adj_M(self, stacked_slices_nda_vec):

        ## Allocate memory
        A_adj_M_y = np.zeros(self._N_voxels_HR_volume)

        ## Define index for first voxel of first slice within array
        i_min = 0

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            ## Get number of voxels of each slice in current stack
            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for j in range(0, stack.get_number_of_slices()):

                ## Define index for last voxel to specify current slice (exlusive)
                i_max = i_min + N_slice_voxels

                ## Get current slice
                slice_k = slices[j]

                ## Extract 1D corresponding to current slice and convert it to itk.Object
                slice_itk = self.get_itk_image_from_array_vec( stacked_slices_nda_vec[i_min:i_max], slice_k.itk )

                ## Apply A_k' M_k on current slice
                Ak_adj_Mk_slice_itk = self.Ak_adj_Mk(slice_itk, slice_k)
                Ak_adj_Mk_slice_nda_vec = self._itk2np.GetArrayFromImage(Ak_adj_Mk_slice_itk).flatten()

                ## Add contribution
                A_adj_M_y = A_adj_M_y + Ak_adj_Mk_slice_nda_vec

                ## Define index for first voxel to specify subsequent slice (inclusive)
                i_min = i_max

        return A_adj_M_y


    ## Convert numpy data array (vector format) back to itk.Image object
    #  \param[in] HR_nda_vec HR data as 1D array
    #  \return HR volume with intensities according to HR_nda_vec as itk.Image object
    def get_itk_image_from_array_vec(self, nda_vec, image_itk_ref):
        
        shape_nda = np.array(image_itk_ref.GetLargestPossibleRegion().GetSize())[::-1]

        image_itk = self._itk2np.GetImageFromArray( nda_vec.reshape(shape_nda) )
        image_itk.SetOrigin( image_itk_ref.GetOrigin() )
        image_itk.SetSpacing( image_itk_ref.GetSpacing() )
        image_itk.SetDirection( image_itk_ref.GetDirection() )

        return image_itk


    ## Update internal Oriented and Adjoint Oriented Gaussian Interpolate Image
    #  Filter parameters. Hence, update combined Downsample and Blur Operator
    #  according to the relative position between slice and HR volume.
    #  \param[in] slice Slice object
    def _update_oriented_adjoint_oriented_Gaussian_image_filters(self, slice):
        ## Get variance covariance matrix representing Gaussian blurring in HR volume coordinates
        Cov_HR_coord = self._psf.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates( slice, self._HR_volume )

        ## Update parameters of forward operator A
        self._filter_oriented_Gaussian_interpolator.SetCovariance( Cov_HR_coord.flatten() )
        self._filter_oriented_Gaussian.SetOutputParametersFromImage( slice.itk )
        
        ## Update parameters of backward/adjoint operator A'
        self._filter_adjoint_oriented_Gaussian.SetCovariance( Cov_HR_coord.flatten() )


