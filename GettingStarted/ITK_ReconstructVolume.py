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
import FirstEstimateOfHRVolume as efhrv


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

            self._alpha_cut = 3     # Cut-off distance for blurring filters
            self._alpha = 0.01      # Regularization parameter

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

            ## Create PyBuffer object for conversion between NumPy arrays and ITK images
            self._itk2np = itk.PyBuffer[image_type]


            ## Extract information ready to use for itk image conversion operations
            self._HR_shape_nda = sitk.GetArrayFromImage( self._HR_volume.sitk ).shape
            self._HR_origin_itk = self._HR_volume.sitk.GetOrigin()
            self._HR_spacing_itk = self._HR_volume.sitk.GetSpacing()
            self._HR_direction_itk = sitkh.get_itk_direction_from_sitk_image( self._HR_volume.sitk )


    ## Get current estimate of HR volume
    #  \return current estimate of HR volume, instance of Stack
    def get_HR_volume(self):
        return self._HR_volume


    ## Set cut-off distance
    #  \param[in] alpha_cut scalar value
    def set_alpha_cut(self, alpha_cut):
        self._alpha_cut = alpha_cut

        ## Update cut-off distance for both image filters
        self._filter_oriented_Gaussian_interpolator.SetAlpha(alpha_cut)
        self._filter_adjoint_oriented_Gaussian.SetAlpha(alpha_cut)


    ## Get cut-off distance
    #  \return scalar value
    def get_alpha_cut(self):
        return self._alpha_cut

    ## Set regularization parameter
    #  \param[in] alpha regularization parameter, scalar
    def set_alpha(self, alpha):
        self._alpha = alpha


    ## Get value of chosen regularization parameter
    #  \return regularization parameter, scalar
    def get_alpha(self):
        return self._alpha


    ## Run reconstruction algorithm 
    def run_reconstruction(self):

        ## Compute required variables prior the optimization step
        HR_nda = sitk.GetArrayFromImage(self._HR_volume.sitk)
        sum_ATy_itk = self._sum_ATy()
        op_sum_ATy_itk = self._op(sum_ATy_itk, self._alpha)

        ## Define function handles for optimization
        f_TK0        = lambda x: self._TK0_SPD(x, sum_ATy_itk, self._alpha)
        f_TK0_grad   = lambda x: self._TK0_SPD_grad(x, op_sum_ATy_itk, self._alpha)
        f_TK0_hess_p = None

        ## Compute TK0 solution
        [HR_volume_itk, res_minimizer_TK0] \
            = self._get_reconstruction(fun=f_TK0, jac=f_TK0_grad, hessp=f_TK0_hess_p, x0=HR_nda.flatten(), info_title="TK0")

        ## Update member attribute
        self._HR_volume.itk = HR_volume_itk
        self._HR_volume.sitk = sitkh.convert_itk_to_sitk_image( HR_volume_itk )


    ## Use scipy.optimize.minimize to get an approximate solution
    #  \param[in] fun   objective function to minimize, returns scalar value
    #  \param[in] jac   jacobian of objective function, returns vector array
    #  \param[in] hessp hessian matrix of objective function applied on point p, returns vector array
    #  \param[in] x0    initial value for optimization, vector array
    #  \param[in] info_title determines which title is used to print information (optional)
    #  \return data array of reconstruction
    #  \return output of scipy.optimize.minimize function
    def _get_reconstruction(self, fun, jac, hessp, x0, info_title=False):
        iter_max = 20      # maximum number of iterations for solver
        tol = 1e-8          # tolerance for solver

        show_disp = True

        ## Provide bounds for optimization, i.e. intensities >= 0
        bounds = [[0,None]]*x0.size

        ## Start timing
        t0 = time.clock()

        ## Find approximate solution
        ## Look at http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        # res = minimize(method='Powell',     fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max})

        # res = minimize(method='L-BFGS-B',   fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac)
        # res = minimize(method='CG',         fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac)
        # res = minimize(method='BFGS',       fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac)

        ## Take incredibly long.
        # res = minimize(method='Newton-CG',  fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac, hessp=hessp)
        # res = minimize(method='trust-ncg',  fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac, hessp=hessp)

        ## Find approximate solution using bounds
        res = minimize(method='L-BFGS-B',   fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max, 'disp': show_disp}, jac=jac, bounds=bounds)
        # res = minimize(method='TNC',   fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max, 'disp': show_disp}, jac=jac, bounds=bounds)


        ## Stop timing
        time_elapsed = time.clock() - t0

        ## Print optimizer status
        if info_title is not None:
            self._print_status_optimizer(res, time_elapsed, info_title)

        ## Convert back to itk.Image object
        HR_volume_itk = self._get_HR_image_from_array_vec(res.x)        

        return [HR_volume_itk, res]


    ## Print information stored in the result variable obtained from
    #  scipy.optimize.minimize.
    #  \param[in] res output from scipy.optimize.minimize
    #  \param[in] time_elapsed measured time via time.clock() (optional)
    #  \param[in] title title printed on the screen for subsequent information
    def _print_status_optimizer(self, res, time_elapsed=None, title="Overview"):
        print("Result optimization: %s" %(title))
        print("\t%s" % (res.message))
        print("\tCurrent value of objective function: %s" % (res.fun))
        try:
            print("\tIterations: %s" % (res.nit))
        except:
            None
        try:
            print("\tFunction evaluations: %s" % (res.nfev))
        except:
            None
        try:
            print("\tGradient evaluations: %s" % (res.njev))
        except:
            None
        try:
            print("\tHessian evaluations: %s" % (res.nhev))
        except:
            None

        if time_elapsed is not None:
            print("\tElapsed time for optimization: %s seconds" %(time_elapsed))



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
        self._filter_adjoint_oriented_Gaussian.UpdateLargestPossibleRegion()
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
    #  \return sum of all forward an back projected operations of image
    #       in HR space as itk.Image object
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
                ATA_x = self._ATA(HR_volume_itk)

                ## Add contribution
                adder.SetInput2(ATA_x)
                adder.Update()

                sum_ATAx = adder.GetOutput()
                sum_ATAx.DisconnectPipeline()

                ## Prepare for next cycle
                adder.SetInput1(sum_ATAx)

        return sum_ATAx


    ## Compute sum_{k=1}^K A_k' y_k for all slices y_k.
    #  \return sum of all back projected slices
    #       in HR space as itk.Image object
    def _sum_ATy(self):

        ## Create image adder
        adder = itk.AddImageFilter[image_type, image_type, image_type].New()
        ## This will copy the buffer of input1 to the output and remove the need to allocate a new output
        adder.InPlaceOn() 

        slice0 = self._stacks[0].get_slice(0)

        ## Update A_k and A_k' based on position of slice
        self._update_oriented_adjoint_oriented_Gaussian_image_filters(slice0)
        ATy = self._AT(slice0.itk)

        ## Duplicate first slice of first stack with zero image buffer
        duplicator = itk.ImageDuplicator[image_type].New()
        duplicator.SetInputImage(ATy)
        duplicator.Update()

        sum_ATy = duplicator.GetOutput()
        sum_ATy.DisconnectPipeline()
        sum_ATy.FillBuffer(0.0)

        ## Insert zero image for image adder
        adder.SetInput1(sum_ATy)


        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            for j in range(0, stack.get_number_of_slices()):

                slice = slices[j]
                # sitkh.show_itk_image(slice.itk,title="slice")

                ## Update A_k and A_k' based on position of slice
                self._update_oriented_adjoint_oriented_Gaussian_image_filters(slice)

                ## Perform A_k'A_k(x) 
                AT_y = self._AT(slice.itk)

                ## Add contribution
                adder.SetInput2(AT_y)
                adder.Update()

                sum_ATy = adder.GetOutput()
                sum_ATy.DisconnectPipeline()

                ## Prepare for next cycle
                adder.SetInput1(sum_ATy)

        return sum_ATy


    ## Convert HR data array (vector format) back to itk.Image object
    #  \param[in] HR_nda_vec HR data as 1D array
    #  \return HR volume with intensities according to HR_nda_vec
    def _get_HR_image_from_array_vec(self, HR_nda_vec):
        
        ## Create ITK image
        image_itk = self._itk2np.GetImageFromArray( HR_nda_vec.reshape( self._HR_shape_nda ) ) 

        image_itk.SetOrigin(self._HR_origin_itk)
        image_itk.SetSpacing(self._HR_spacing_itk)
        image_itk.SetDirection(self._HR_direction_itk)

        image_itk.DisconnectPipeline()

        return image_itk


    ## Compute I0 + const*I1 with I0 and I1 being itk.Image objects occupying
    #  the same physical space
    #  \param[in] image0_itk first image, itk.Image object
    #  \param[in] const constant to multiply second image, scalar
    #  \param[in] image1_itk second image, itk.Image object
    #  \return image0_itk + const*image1_itk as itk.Image object
    def _add_amplified_image(self, image0_itk, const, image1_itk):

        ## Create image adder and multiplier
        adder = itk.AddImageFilter[image_type, image_type, image_type].New()
        multiplier = itk.MultiplyImageFilter[image_type, image_type, image_type].New()

        ## compute const*image1_itk
        multiplier.SetInput( image1_itk )
        multiplier.SetConstant( const )

        ## compute image0_itk + const*image1_itk
        adder.SetInput1( image0_itk )
        adder.SetInput2( multiplier.GetOutput() )
        adder.Update()

        res = adder.GetOutput()
        res.DisconnectPipeline()

        return res


    ## Compute 
    #       op(x) := ( sum_k [A_k' A_k] + alpha )*x
    #                sum_k [A_k' A_k x] + alpha*x
    #  \param[in] image_itk image which acts as x, itk.Image object
    #  \param[in] alpha regularization parameter, scalar
    #  \return op(x) = ( sum_k [A_k' A_k] + alpha )*x as itk.Image object
    def _op(self, image_itk, alpha):

        ## Compute sum_k [A_k' A_k x]
        sum_ATAx_itk = self._sum_ATA(image_itk)        

        ## Compute sum_k [A_k' A_k x] + alpha*x
        return self._add_amplified_image(sum_ATAx_itk, alpha, image_itk)


    ## Compute cost function with SPD matrix
    #       J0(x) = || sum_k [A_k' (A_k x - y_k)] + alpha*x ||^2
    #             = || (sum_k [A_k' A_k ] + alpha)x - sum_k A_k' y_k || ^2
    def _TK0_SPD(self, HR_nda_vec, sum_ATy_itk, alpha):

        ## Convert HR data array back to itk.Image object
        x_itk = self._get_HR_image_from_array_vec(HR_nda_vec)

        ## Compute op(x) = sum_k [A_k' A_k x] + alpha*x
        op_x_itk = self._op(x_itk, alpha)

        ## Compute sum_k [A_k' A_k x] + alpha*x - sum_k A_k' y_k 
        J0_image_itk =  self._add_amplified_image(op_x_itk, -1, sum_ATy_itk)

        ## J0 = || sum_k [A_k' A_k x] + alpha*x - sum_k A_k' y_k ||^2
        J0_nda = self._itk2np.GetArrayFromImage(J0_image_itk)

        return np.sum((J0_nda)**2)


    ## Compute gradient of cost function with SPD matrix
    #       grad J0(x) = (sum_k [A_k' A_k] + alpha) ( (sum_k [A_k' A_k] + alpha)x - sum_k A_k' y_k  )
    def _TK0_SPD_grad(self, HR_nda_vec, op_sum_ATy_itk, alpha):

        ## Convert HR data array back to itk.Image object
        x_itk = self._get_HR_image_from_array_vec(HR_nda_vec)

        ## Compute op(x) := sum_k [A_k' A_k x] + alpha*x
        op_x_itk = self._op(x_itk, alpha)

        ## Compute op(op(x)) = sum_k [A_k' A_k op(x)] + alpha*op(x)
        op_op_x_itk = self._op(op_x_itk, alpha)

        ## Compute grad J0 in image space
        grad_J0_image_itk = self._add_amplified_image(op_op_x_itk, -1, op_sum_ATy_itk)

        ## return grad J0 in voxel space
        return self._itk2np.GetArrayFromImage(grad_J0_image_itk).flatten()



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
        filename_out = "pig"

    ## Data of GettingStarted folder
    else:
        dir_input = "data/"
        filenames = [
            "FetalBrain_stack0_registered",
            "FetalBrain_stack1_registered",
            "FetalBrain_stack2_registered"
            ]
        filename_HR_volume = "FetalBrain_reconstruction_4stacks"
        filename_out = "fetalbrain"

    ## Output folder
    dir_output = "results/recon/"

    ## Prepare output directory
    reconstruction_manager = rm.ReconstructionManager(dir_output)

    ## Read input data
    reconstruction_manager.read_input_data(dir_input, filenames)

    ## Compute first estimate of HR volume (averaged volume)
    reconstruction_manager.compute_first_estimate_of_HR_volume(use_in_plane_registration=False)    
    HR_volume = reconstruction_manager.get_HR_volume()

    ## Copy initial HR volume for comparison later on
    HR_init_sitk = sitk.Image(HR_volume.sitk)

    # HR_volume.show()

    ## HR volume reconstruction obtained from Kainz toolkit
    HR_volume_Kainz = st.Stack.from_nifti(dir_input,filename_HR_volume)

    ## Initialize optimizer with current state of motion estimation + guess of HR volume
    MyOptimizer = Optimize(reconstruction_manager.get_stacks(), HR_volume)

    ## Perform reconstruction
    print("\n--- Run reconstruction algorithm ---")
    MyOptimizer.run_reconstruction()

    ## Get reconstruction result
    recon = MyOptimizer.get_HR_volume()

    sitkh.show_sitk_image(HR_init_sitk, overlay_sitk=recon.sitk, title="HR_init+recon")


    sitk.WriteImage(recon.sitk,dir_output+filename_out+"_recon.nii.gz")
    sitk.WriteImage(HR_volume.sitk,dir_output+filename_out+"_init.nii.gz")


    # stacks = reconstruction_manager.get_stacks()
    # N_stacks = len(stacks)
    # stack = stacks[1]
    # slices = stack.get_slices()
    # sitkh.show_sitk_image(HR_volume.sitk)


    # stacks[1].get_slice(0).write(directory=dir_output, filename="slice")
    # HR_volume.write(directory=dir_output, filename="HR_volume")




