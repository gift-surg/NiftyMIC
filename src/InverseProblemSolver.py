## \file InverseProblemSolver.py
#  \brief Implementation to get an approximate solution of the inverse problem 
#  \f$ y_k = A_k x \f$ for each slice \f$ y_k,\,k=1,\dots,K \f$
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date March 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
import time  
from scipy.optimize import minimize
from scipy import ndimage
import matplotlib.pyplot as plt

## Import modules from src-folder
import SimpleITKHelper as sitkh
import PSF as psf


## Pixel type of used 3D ITK image
pixel_type = itk.D

## ITK image type 
image_type = itk.Image[pixel_type, 3]


## This class is used to compute an approximate solution of the HR volume 
#  \f$ x \f$  as defined by the inverse problem \f$ y_k = A_k x \f$ for every 
#  slice \f$ y_k,\,k=1,\dots,K \f$ in the regularized form
#  \f[ 
#       \sum_{k=1}^K \frac{1}{2} \Vert y_k - A_k x \Vert_{\ell^2}^2 + \alpha\,\Psi(x) 
#       = \sum_{k=1}^K \frac{1}{2} \Vert y_k - A_k x \Vert_{\ell^2}^2 + \alpha\, \frac{1}{2}\Vert Gx \Vert_{\ell^2}^2
#       \rightarrow \min_x  
#  \f]
#  by reformulating it as
#  \f[
#       J(x) = \frac{1}{2} \Vert \Big(\sum_k A_k^* (A_k x - y_k) \Big) + \alpha\, G^*G x \Vert_{\ell^2}^2
#             = \frac{1}{2} \Vert \Big(\sum_k A_k^* M_k A_k + \alpha\, G^*G \Big)x - \sum_k A_k^* M_k y_k \Vert_{\ell^2}^2
#       \rightarrow \min_x 
#  \f]
#  where \f$A_k=D_k B_k\f$ denotes the combined blurring and downsampling 
#  operation and \f$G\f$ represents either the identity matrix \f$I\f$ 
#  (zero-order Tikhonov) or the gradient \f$ \nabla \f$ (first-order Tikhonov).
#  \see \p itkAdjointOrientedGaussianInterpolateImageFilter of \p ITK
#  \see \p itOrientedGaussianInterpolateImageFunction of \p ITK
#  \warning HACK: Append slices as itk image on each object Slice
class InverseProblemSolver:

    ## Constructor
    #  \param[in] stacks list of Stack objects containing all stacks used for the reconstruction
    #  \param[in] HR_volume Stack object containing the current estimate of the HR volume (used as initial value + space definition)
    def __init__(self, stacks, HR_volume):

        ## Initialize variables
        self._stacks = stacks
        self._HR_volume = HR_volume
        self._N_stacks = len(stacks)

        ## Used for PSF modelling
        self._psf = psf.PSF()

        ## Cut-off distance for Gaussian blurring filter
        self._alpha_cut = 3     

        ## Settings for optimizer
        self._alpha = 0.1      # Regularization parameter
        self._iter_max = 20     # Maximum iteration steps
        self._reg_type = 'TK0'  # Either Tikhonov zero- or first-order

        ## Append itk objects
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

        ## Define dictionary to choose between two possible computations
        #  of the differential operator D'D
        self._DTD = {
            "Laplace"           :   self._DTD_laplacian,
            "FiniteDifference"  :   self._DTD_finite_diff
        }
        self._DTD_comp_type = "Laplace" #default value


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


    ## Set maximum number of iterations for minimizer
    #  \param[in] iter_max number of maximum iterations, scalar
    def set_iter_max(self, iter_max):
        self._iter_max = iter_max


    ## Get chosen value of maximum number of iterations for minimizer
    #  \return maximum number of iterations set for minimizer, scalar
    def get_iter_max(self):
        return self._iter_max


    ## Set type or regularization. It can be either 'TK0' or 'TK1'
    #  \param[in] reg_type Either 'TK0' or 'TK1', string
    def set_regularization_type(self, reg_type):
        if reg_type not in ["TK0", "TK1"]:
            raise ValueError("Error: regularization type can only be either 'TK0' or 'TK1'")

        self._reg_type = reg_type


    ## Get chosen type of regularization.
    #  \return regularization type as string
    def get_regularization_type(self):
        return self._reg_type


    ## The differential operator \f$ D^*D \f$ for TK1 regularization can be computed
    #  via either a sequence of finited differences in each spatial 
    #  direction or directly via a Laplacian stencil
    #  \param[in] DTD_comp_type "Laplacian" or "FiniteDifference"
    def set_DTD_computation_type(self, DTD_comp_type):

        if DTD_comp_type not in ["Laplace", "FiniteDifference"]:
            raise ValueError("Error: D'D computation type can only be either 'Laplace' or 'FiniteDifference'")

        else:
            self._DTD_comp_type = DTD_comp_type


    ## Get chosen type of computation for differential operation D'D
    #  \return type of \f$ D^*D \f$ computation, string
    def get_DTD_computation_type(self):
        return self._DTD_comp_type


    ## Run reconstruction algorithm 
    def run_reconstruction(self):

        ## Compute required variables prior the optimization step
        HR_nda = sitk.GetArrayFromImage(self._HR_volume.sitk)
        sum_ATMy_itk = self._sum_ATMy()

        ## TK0-regularization
        if self._reg_type in ["TK0"]:
            print("Chosen regularization type: zero-order Tikhonov")
            print("Regularization paramter = " + str(self._alpha))
            print("Maximum number of iterations is set to " + str(self._iter_max))

            ## Provide constant variable for optimization
            op0_sum_ATMy_itk = self._op0(sum_ATMy_itk, self._alpha)

            ## Define function handles for optimization
            f        = lambda x: self._TK0_SPD(x, sum_ATMy_itk, self._alpha)
            f_grad   = lambda x: self._TK0_SPD_grad(x, op0_sum_ATMy_itk, self._alpha)
            f_hess_p = None


        ## TK1-regularization
        elif self._reg_type in ["TK1"]:
            
            ## Compute kernels for differentiation in image space, i.e. including scaling
            spacing = self._HR_volume.sitk.GetSpacing()

            ## DTD is computed direclty via Laplace stencil
            if self._DTD_comp_type in ["Laplace"]:
                print("Chosen regularization type: first-order Tikhonov via Laplace stencil")
                print("Regularization paramter = " + str(self._alpha))
                print("Maximum number of iterations is set to " + str(self._iter_max))
                
                # Laplace kernel
                self._kernel_L = self._get_laplacian_kernel() / (spacing[0]*spacing[0])

                # Set finite difference kernels to None
                self._kernel_Dx     = None
                self._kernel_Dy     = None
                self._kernel_Dz     = None
                self._kernel_DTx    = None
                self._kernel_DTy    = None
                self._kernel_DTz    = None

            ## DTD is computed as sequence of forward and backward operators
            else:
                print("Chosen regularization type: first-order Tikhonov via forward/backward finite differences")
                print("Regularization paramter = " + str(self._alpha))
                print("Maximum number of iterations is set to " + str(self._iter_max))

                # Forward difference kernels
                kernel_Dxf = self._get_forward_diff_x_kernel() / spacing[0]
                kernel_Dyf = self._get_forward_diff_y_kernel() / spacing[0]
                kernel_Dzf = self._get_forward_diff_z_kernel() / spacing[0]

                # Backward difference kernels
                kernel_Dxb = self._get_backward_diff_x_kernel() / spacing[0]
                kernel_Dyb = self._get_backward_diff_y_kernel() / spacing[0]
                kernel_Dzb = self._get_backward_diff_z_kernel() / spacing[0]

                # Finite difference kernels
                self._kernel_Dx = kernel_Dxf
                self._kernel_Dy = kernel_Dyf
                self._kernel_Dz = kernel_Dzf
                self._kernel_DTx = -kernel_Dxb
                self._kernel_DTy = -kernel_Dyb
                self._kernel_DTz = -kernel_Dzb

                # Set Laplace kernel to None
                self._kernel_L   = None

            ## Provide constant variable for optimization
            op1_sum_ATMy_itk = self._op1(sum_ATMy_itk, HR_nda.flatten(), self._alpha)
            
            ## Define function handles for optimization
            f        = lambda x: self._TK1_SPD(x, sum_ATMy_itk, self._alpha)
            f_grad   = lambda x: self._TK1_SPD_grad(x, op1_sum_ATMy_itk, self._alpha)
            f_hess_p = None

        ## Compute approximate solution
        [HR_volume_itk, res_minimizer] \
            = self._get_reconstruction(fun=f, jac=f_grad, hessp=f_hess_p, x0=HR_nda.flatten(), info_title=False)

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
        iter_max = self._iter_max      # maximum number of iterations for minimizer
        tol = 1e-8                     # tolerance for minimizer

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

        ## Constrained optimization
        res = minimize(method='L-BFGS-B',   fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max, 'disp': show_disp}, jac=jac, bounds=bounds)
        # res = minimize(method='TNC',        fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max, 'disp': show_disp}, jac=jac, bounds=bounds) #useless; tnc: Maximum number of function evaluations reached
        # res = minimize(method='SLSQP',      fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max, 'disp': show_disp}, jac=jac, bounds=bounds) #crashes python


        ## Stop timing
        time_elapsed = time.clock() - t0

        ## Print optimizer status
        if info_title is not False:
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
    # \warning HACK
    def _append_itk_object_on_slices_of_stacks(self):

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            for j in range(0, stack.get_number_of_slices()):
                slice = slices[j]

                slice_sitk      = sitk.Image(slice.sitk)
                slice_sitk_mask = sitk.Image(slice.sitk_mask)

                # ## Only use segmented part of each slice (if available)
                # if slice.sitk_mask is not None:
                #     slice_masked_sitk = sitk.Cast(slice.sitk_mask, slice.sitk.GetPixelIDValue()) * slice.sitk
                # else:
                #     slice_masked_sitk = sitk.Image(slice.sitk)

                slice.itk       = sitkh.convert_sitk_to_itk_image(slice_sitk)
                slice.itk_mask  = sitkh.convert_sitk_to_itk_image(slice_sitk_mask)
                

    ## Update internal Oriented and Adjoint Oriented Gaussian Interpolate Image Filter parameters. Hence, update combined Downsample and Blur Operator
    #  according to the relative position between slice and HR volume
    #  \param[in] slice Slice object
    def _update_oriented_adjoint_oriented_Gaussian_image_filters(self, slice):
        ## Get variance covariance matrix representing Gaussian blurring in HR volume coordinates
        Cov_HR_coord = self._psf.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates( slice, self._HR_volume )

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
        

    ## Perform forward operation on HR image, i.e. \f$y = DBx =: Ax \f$ with \f$D\f$  and \f$ B \f$ being 
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


    ## Perform backward operation on LR image, i.e. \f$z = B^*D^*My = A^*My \f$ with \f$ D^* \f$ and \f$ B^* \f$ being 
    #  the adjoint downsampling and blurring operator, respectively.
    #  \param[in] slice_itk LR image as itk.Image object
    #  \param[in] mask_itk mask of LR image as itk.Image object
    #  \return image in HR space as itk.Image object after performed backward operation
    def _ATM(self, slice_itk, mask_itk):

        multiplier = itk.MultiplyImageFilter[image_type, image_type, image_type].New()

        ## compute M y_k
        multiplier.SetInput1( mask_itk )
        multiplier.SetInput2( slice_itk  )
        multiplier.Update()

        self._filter_adjoint_oriented_Gaussian.SetInput( multiplier.GetOutput() )
        self._filter_adjoint_oriented_Gaussian.UpdateLargestPossibleRegion()
        self._filter_adjoint_oriented_Gaussian.Update()

        HR_volume_itk = self._filter_adjoint_oriented_Gaussian.GetOutput()
        HR_volume_itk.DisconnectPipeline()

        return HR_volume_itk


    ## Perform the operation \f$ A^* MAx \f$ with \f$ A = DB \f$, i.e. the combination of
    #  downsampling \f$D\f$ and blurring \f$B\f$, and the masking operator \f$M\f$.
    #  \param[in] HR_volume_itk HR image as itk.Image object
    #  \param[in] mask_slice_itk mask of slice as itk.Image object
    #  \return image in HR space as itk.Image object after performed forward 
    #          and backward operation
    def _ATMA(self, HR_volume_itk, mask_slice_itk):
        return self._ATM( self._A(HR_volume_itk), mask_slice_itk )


    ## Compute \f$ \sum_{k=1}^K A_k^* M_k A_k x \f$
    #  \param[in] HR_volume_itk HR image as itk.Image object
    #  \return sum of all forward an back projected operations of image
    #       in HR space as itk.Image object
    def _sum_ATMA(self, HR_volume_itk):

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

                ## Perform A_k'MA_k(x) 
                ATMA_x = self._ATMA(HR_volume_itk, slice.itk_mask)

                ## Add contribution
                adder.SetInput2(ATMA_x)
                adder.Update()

                sum_ATMAx = adder.GetOutput()
                sum_ATMAx.DisconnectPipeline()

                ## Prepare for next cycle
                adder.SetInput1(sum_ATMAx)

        return sum_ATMAx


    ## Compute \f$ \sum_{k=1}^K A_k^* M_k y_k \f$ for all slices \f$ y_k \f$.
    #  \return sum of all back projected slices
    #       in HR space as itk.Image object
    def _sum_ATMy(self):

        ## Create image adder
        adder = itk.AddImageFilter[image_type, image_type, image_type].New()
        ## This will copy the buffer of input1 to the output and remove the need to allocate a new output
        adder.InPlaceOn() 

        slice0 = self._stacks[0].get_slice(0)

        ## Update A_k and A_k' based on position of slice
        self._update_oriented_adjoint_oriented_Gaussian_image_filters(slice0)
        ATMy = self._ATM(slice0.itk, slice0.itk_mask)

        ## Duplicate first slice of first stack with zero image buffer
        duplicator = itk.ImageDuplicator[image_type].New()
        duplicator.SetInputImage(ATMy)
        duplicator.Update()

        sum_ATMy = duplicator.GetOutput()
        sum_ATMy.DisconnectPipeline()
        sum_ATMy.FillBuffer(0.0)

        ## Insert zero image for image adder
        adder.SetInput1(sum_ATMy)


        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            for j in range(0, stack.get_number_of_slices()):

                slice = slices[j]
                # sitkh.show_itk_image(slice.itk,title="slice")

                ## Update A_k and A_k' based on position of slice
                self._update_oriented_adjoint_oriented_Gaussian_image_filters(slice)

                ## Perform A_k'M_kA_k(x) 
                ATM_y = self._ATM(slice.itk, slice.itk_mask)

                ## Add contribution
                adder.SetInput2(ATM_y)
                adder.Update()

                sum_ATMy = adder.GetOutput()
                sum_ATMy.DisconnectPipeline()

                ## Prepare for next cycle
                adder.SetInput1(sum_ATMy)

        return sum_ATMy


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


    """
    TK0-regularization functions
    """
    ## Compute
    #  \f$
    #         op_0(x):= \Big( \sum_k A_k^* M_k A_k + \alpha \Big) x
    #                 = \sum_k A_k^* M_k A_k x + \alpha x
    #  \f$
    #  \param[in] image_itk image which acts as x, itk.Image object
    #  \param[in] alpha regularization parameter, scalar
    #  \return op0(x) as itk.Image object
    def _op0(self, image_itk, alpha):

        ## Compute sum_k [A_k' M_k A_k x]
        sum_ATMAx_itk = self._sum_ATMA(image_itk)

        ## Compute sum_k [A_k' M_k A_k x] + alpha*x
        return self._add_amplified_image(sum_ATMAx_itk, alpha, image_itk)


    ## Compute TK0 cost function with SPD matrix
    # \f[   
    #       J_0(x) = \frac{1}{2} \Vert \Big(\sum_k A_k^* M_k (A_k x -y_k) \Big) + \alpha x \Vert_{\ell^2}^2
    #             = \frac{1}{2} \Vert \Big(\sum_k A_k^* M_k A_k + \alpha \Big)x - \sum_k A_k^* M_k y_k \Vert_{\ell^2}^2
    # \f]
    #  \param[in] HR_nda_vec data array of HR image, 1D array shape
    #  \param[in] sum_ATMy_itk output of _sum_ATMy
    #  \param[in] alpha regularization parameter
    #  \return J0(x) as scalar value
    def _TK0_SPD(self, HR_nda_vec, sum_ATMy_itk, alpha):

        ## Convert HR data array back to itk.Image object
        x_itk = self._get_HR_image_from_array_vec(HR_nda_vec)

        ## Compute op0(x) = sum_k [A_k' M_k A_k x] + alpha*x
        op0_x_itk = self._op0(x_itk, alpha)

        ## Compute sum_k [A_k' M_k A_k x] + alpha*x - sum_k A_k' M_k y_k 
        J0_image_itk =  self._add_amplified_image(op0_x_itk, -1, sum_ATMy_itk)

        ## J0 = 0.5*|| sum_k [A_k' M_k A_k x] + alpha*x - sum_k A_k' M_k y_k ||^2
        J0_nda = self._itk2np.GetArrayFromImage(J0_image_itk)

        return 0.5*np.sum( J0_nda**2 )


    ## Compute gradient of TK0 cost function with SPD matrix
     # \f[
    #       \nabla J_0(x) =  \Big(\sum_k A_k^* M_k A_k + \alpha \Big)
    #                        \Big(\big(\sum_k A_k^* M_k A_k + \alpha \big)x - \sum_k A_k^* M_k y_k \Big)
    # \f]
    #  \param[in] HR_nda_vec data array of HR image, 1D array shape
    #  \param[in] op0_sum_ATMy_itk output of _op0(sum_ATMy)
    #  \param[in] alpha regularization parameter
    #  \return grad J0(x) in voxel space as 1D data array
    def _TK0_SPD_grad(self, HR_nda_vec, op0_sum_ATMy_itk, alpha):

        ## Convert HR data array back to itk.Image object
        x_itk = self._get_HR_image_from_array_vec(HR_nda_vec)

        ## Compute op0(x) = sum_k [A_k' M_k A_k x] + alpha*x
        op0_x_itk = self._op0(x_itk, alpha)

        ## Compute op0(op0(x)) = sum_k [A_k' M_k A_k op0(x)] + alpha*op0(x)
        op0_op0_x_itk = self._op0(op0_x_itk, alpha)

        ## Compute grad J0 in image space
        grad_J0_image_itk = self._add_amplified_image(op0_op0_x_itk, -1, op0_sum_ATMy_itk)

        ## Return grad J0 in voxel space
        return self._itk2np.GetArrayFromImage(grad_J0_image_itk).flatten()


    """
    TK1-regularization functions
    """
    ## Compute forward difference quotient in x-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(nda) to differentiate image
    #  \return kernel for 3-dimensional differentiation in x
    def _get_forward_diff_x_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,1,2))
        kernel[:] = np.array([1,-1])

        return kernel


    ## Compute backward difference quotient in x-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(nda) to differentiate image
    #  \return kernel for 3-dimensional differentiation in x
    def _get_backward_diff_x_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,1,3))
        kernel[:] = np.array([0,1,-1])

        return kernel


    ## Compute forward difference quotient in y-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(kernel, nda) to differentiate image
    #  \return kernel kernel for 3-dimensional differentiation in y
    def _get_forward_diff_y_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,2,1))
        kernel[:] = np.array([[1],[-1]])

        return kernel


    ## Compute backward difference quotient in y-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(kernel, nda) to differentiate image
    #  \return kernel kernel for 3-dimensional differentiation in y
    def _get_backward_diff_y_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,3,1))
        kernel[:] = np.array([[0],[1],[-1]])

        return kernel


    ## Compute forward difference quotient in z-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(kernel, nda) to differentiate image
    #  \return kernel kernel for 3-dimensional differentiation in z
    def _get_forward_diff_z_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((2,1,1))
        kernel[:] = np.array([[[1]],[[-1]]])

        return kernel


    ## Compute backward difference quotient in y-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(kernel, nda) to differentiate image
    #  \return kernel kernel for 3-dimensional differentiation in z
    def _get_backward_diff_z_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((3,1,1))
        kernel[:] = np.array([[[0]],[[1]],[[-1]]])

        return kernel


    ## Compute Laplacian kernel to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(kernel, nda) to differentiate image
    #  \param[in] self spatial dimension
    #  \return kernel kernel for Laplacian operation
    def _get_laplacian_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((3,3,3))
        kernel[0,1,1] = 1
        kernel[1,:,:] = np.array([[0, 1, 0],[1, -6, 1],[0, 1, 0]])
        kernel[2,1,1] = 1
            
        return kernel


    ## Compute \f$ D^*Dx \f$ directly via Laplacian stencil 
    #  Chosen kernels already incorporate correct scaling to transform 
    #  resulting data array back directly to image space.
    #  \param[in] nda data array of image
    #  \return \f$ D^*Dx \f$ as itk.Image object
    def _DTD_laplacian(self, nda):

        ## DTDx via Laplacian stencil
        DTD = self._convolve(nda, self._kernel_L)
        
        return self._get_HR_image_from_array_vec( DTD )


    ## Compute \f$ D^*Dx \f$ via a sequence of forward and backward finite differences.
    #  Chosen kernels already incorporate correct scaling to transform 
    #  resulting data array back directly to image space.
    #  \param[in] nda data array of image
    #  \return \f$ D^*Dx \f$ as itk.Image object
    def _DTD_finite_diff(self, nda):

        ## Forward operation
        Dx = self._convolve(nda, self._kernel_Dx)
        Dy = self._convolve(nda, self._kernel_Dy)
        Dz = self._convolve(nda, self._kernel_Dz)

        ## Adjoint operation
        DTDx = self._convolve(Dx, self._kernel_DTx)
        DTDy = self._convolve(Dy, self._kernel_DTy)
        DTDz = self._convolve(Dz, self._kernel_DTz)
        
        ## DTDx via forward and backward differences
        DTD = (DTDx + DTDy + DTDz)

        return self._get_HR_image_from_array_vec( DTD )


    ## Compute convolution of array based on given kernel via 
    #  scipy.ndimage.convolve with "wrap" boundary conditions.
    #  \param[in] nda data array
    #  \param[in] kernel 
    #  \return data array convolved by given kernel
    def _convolve(self, nda, kernel):
        return ndimage.convolve(nda, kernel, mode='wrap')


    ## Compute
    #  \f$
    #         op_1(x):= \Big( \sum_k A_k^* M_k A_k + \alpha\,D^*D \Big) x
    #                 = \sum_k A_k^* M_k A_k x + \alpha\,D^*D x
    #  \f$
    #  \param[in] image_itk image which acts as x, itk.Image object
    #  \param[in] alpha regularization parameter, scalar
    #  \return op1(x) as itk.Image object
    def _op1(self, image_itk, image_nda_vec, alpha):

        ## Compute sum_k [A_k' M_k A_k x]
        sum_ATMAx_itk = self._sum_ATMA(image_itk)   

        ## Compute \f$ D^*Dx \f$
        DTDx_itk = self._DTD[self._DTD_comp_type](image_nda_vec.reshape(self._HR_shape_nda))     

        ## Compute sum_k [A_k' M_k A_k x] + alpha*D'Dx
        return self._add_amplified_image(sum_ATMAx_itk, alpha, DTDx_itk)


    ## Compute TK1 cost function with SPD matrix
    # \f[
    #       J_1(x) = \frac{1}{2} \Vert \Big(\sum_k A_k^* M_k (A_k x -y_k) \Big) + \alpha\, D^*D x \Vert_{\ell^2}^2
    #             = \frac{1}{2} \Vert \Big(\sum_k A_k^* M_k A_k + \alpha\, D^*D \Big)x - \sum_k A_k^* M_k y_k \Vert_{\ell^2}^2
    # \f]
    #  \param[in] HR_nda_vec data array of HR image, 1D array shape
    #  \param[in] sum_ATMy_itk output of _sum_ATMy
    #  \param[in] alpha regularization parameter
    #  \return J1(x) as scalar value
    def _TK1_SPD(self, HR_nda_vec, sum_ATMy_itk, alpha):

        ## Convert HR data array back to itk.Image object
        x_itk = self._get_HR_image_from_array_vec(HR_nda_vec)

        ## Compute op1(x) = sum_k [A_k' M_k A_k x] + alpha*B'Bx
        op1_x_itk = self._op1(x_itk, HR_nda_vec, alpha)

        ## Compute sum_k [A_k' M_k A_k x] + alpha*B'Bx - sum_k A_k' M_k y_k 
        J1_image_itk =  self._add_amplified_image(op1_x_itk, -1, sum_ATMy_itk)

        ## J1 = 0.5*|| sum_k [A_k' M_k A_k x] + alpha*B'Bx - sum_k A_k' M_k y_k ||^2
        J1_nda = self._itk2np.GetArrayFromImage(J1_image_itk)

        return 0.5*np.sum( J1_nda**2 )


    ## Compute gradient of TK1 cost function with SPD matrix
    # \f[
    #       \nabla J_1(x) =  \Big(\sum_k A_k^* M_k A_k + \alpha\, D^*D \Big)
    #                        \Big(\big(\sum_k A_k^* M_k A_k + \alpha\, D^*D \big)x - \sum_k A_k^* M_k y_k \Big)
    # \f]
    #  \param[in] HR_nda_vec data array of HR image, 1D array shape
    #  \param[in] op1_sum_ATMy_itk output of _op1(sum_ATMy)
    #  \param[in] alpha regularization parameter
    #  \return grad J1(x) in voxel space as 1D data array
    def _TK1_SPD_grad(self, HR_nda_vec, op1_sum_ATMy_itk, alpha):

        ## Convert HR data array back to itk.Image object
        x_itk = self._get_HR_image_from_array_vec(HR_nda_vec)

        ## Compute op1(x) = sum_k [A_k' M_k A_k x] + alpha*B'Bx
        op1_x_itk = self._op1(x_itk, HR_nda_vec, alpha)

        ## Compute op1(op1(x)) = sum_k [A_k' M_k A_k op1(x)] + alpha*op1(x)
        op1_op1_x_itk = self._op1(op1_x_itk, HR_nda_vec, alpha)

        ## Compute grad J1 in image space
        grad_J1_image_itk = self._add_amplified_image(op1_op1_x_itk, -1, op1_sum_ATMy_itk)

        ## Return grad J1 in voxel space
        return self._itk2np.GetArrayFromImage(grad_J1_image_itk).flatten()
