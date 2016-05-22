## \file RegularizationParameterEstimation.py
#  \brief This class serves to estimate the regularization parameter \f$ \alpha \f$
#  in order to minimize
#  \f[
#        \frac{1}{2} \sum_k \Vert M_k y_k - M_k A_k x \Vert_{\ell^2}^2 + \alpha\,\Psi(x) \rightarrow \min_x
#  \f]
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


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
import Stack as st
import InverseProblemSolver as ips


## Pixel type of used 3D ITK image
pixel_type = itk.D

## ITK image type 
image_type = itk.Image[pixel_type, 3]

class RegularizationParameterEstimation:

    ## Constructor
    #  \param[in] stacks list of Stack objects containing all stacks used for the reconstruction
    #  \param[in] HR_volume Stack object containing the current estimate of the HR volume (used as initial value + space definition)
    def __init__(self, stacks, HR_volume):

        ## Initialize variables
        self._stacks = stacks
        self._HR_volume = HR_volume
        self._N_stacks = len(stacks)

        self._alphas = None
        self._residuals = None
        self._Psis = None
        self._regularization_types = None

        self._dir_results = "RegularizationParameterEstimation/"
        os.system("mkdir -p " + self._dir_results)

        """
        Member functions to compute residual and prior term Psi
        """
        ## Used for PSF modelling
        self._psf = psf.PSF()

        ## Cut-off distance for Gaussian blurring filter
        self._alpha_cut = 3     

        ## Allocate and initialize Oriented Gaussian Interpolate Image Filter
        self._filter_oriented_Gaussian_interpolator = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        self._filter_oriented_Gaussian_interpolator.SetAlpha(self._alpha_cut)
        
        self._filter_oriented_Gaussian = itk.ResampleImageFilter[image_type, image_type].New()
        self._filter_oriented_Gaussian.SetInterpolator(self._filter_oriented_Gaussian_interpolator)
        self._filter_oriented_Gaussian.SetDefaultPixelValue( 0.0 )

        ## Create PyBuffer object for conversion between NumPy arrays and ITK images
        self._itk2np = itk.PyBuffer[image_type]

        ## Extract information ready to use for itk image conversion operations
        self._HR_shape_nda = sitk.GetArrayFromImage( self._HR_volume.sitk ).shape
        self._HR_origin_itk = self._HR_volume.sitk.GetOrigin()
        self._HR_spacing_itk = self._HR_volume.sitk.GetSpacing()
        self._HR_direction_itk = sitkh.get_itk_direction_from_sitk_image( self._HR_volume.sitk )

        ## Define dictionary to choose between computations of prior term \f$ \Psi(x) \f$
        self._compute_prior_term = {
            "TK0"   : self._compute_prior_term_TK0,
            "TK1"   : self._compute_prior_term_TK1
        }


    ## Define array of regularization parameters which will be used for
    #  the computation
    #  \param[in] alpha_array numpy array of alpha values
    def set_alpha_array(self, alpha_array):
        self._alphas = alpha_array


    ## Get chosen values for alpha
    #  \return numpy array of alpha values
    def get_alpha_array(self):
        return self._alphas


    ## Set SRR approaches which will be run
    #  \param[in] regularization_types list of strings containing only "TK0", "TK1"
    def set_regularization_types(self, regularization_types):
        self._regularization_types = regularization_types


    ## Get chosen types of SRR
    #  \return list of strings
    def get_regularization_types(self):
        return self._regularization_types


    ## Run all reconstructions for all chosen regularization types and
    #  associated regularization parameters alpha
    def run_reconstructions(self, save_flag=0):

        N_alphas = len(self._alphas)
        N_regularization_types = len(self._regularization_types)

        self._Psis = np.zeros((N_regularization_types, N_alphas))
        self._residuals = np.zeros((N_regularization_types, N_alphas))

        ## Cut-off distance for Gaussian blurring filter
        SRR_alpha_cut = 3 

        # Settings for optimizer
        SRR_iter_max = 10                # Maximum iteration steps
        SRR_DTD_comp_type = "FiniteDifference" # TK1: Either 'Laplace' or 'FiniteDifference'
        SRR_tolerance = 1e-5

        SRR_rho = 1            # Regularization parameter of augmented Lagrangian term (only for TV-L2)
        ADMM_iterations = 5    # Number of performed ADMM iterations
        SRR_ADMM_iterations_output_dir = None

        for i_reg_type in range(0, N_regularization_types):

            ## Read type of regularization
            SRR_approach = self._regularization_types[i_reg_type]

            ## Run 
            for i in range(0, N_alphas):
                print("\n\t--- %s-Regularization (%s/%s): Reconstruction with alpha = %s (%s/%s) ---" %(SRR_approach, i_reg_type+1, N_regularization_types, self._alphas[i], i+1, N_alphas))
                alpha = self._alphas[i]
                HR_volume_init = st.Stack.from_stack(self._HR_volume)

                ## Super-Resolution Reconstruction object
                SRR = ips.InverseProblemSolver(self._stacks, HR_volume_init)

                ## Set current alpha
                SRR.set_alpha(alpha)

                ## Set other values
                SRR.set_alpha_cut(SRR_alpha_cut)
                
                SRR.set_iter_max(SRR_iter_max)
                SRR.set_regularization_type(SRR_approach)
                SRR.set_DTD_computation_type(SRR_DTD_comp_type)
                SRR.set_tolerance(SRR_tolerance)

                SRR.set_rho(SRR_rho)
                SRR.set_ADMM_iterations(ADMM_iterations)
                SRR.set_ADMM_iterations_output_dir(SRR_ADMM_iterations_output_dir)
                
                ## Reconstruct volume for given alpha
                SRR.run_reconstruction()
                HR_volume_alpha = SRR.get_HR_volume()

                ## Write result
                if save_flag:
                    filename =  "RegularizationParameterEstimation"
                    filename += "_" + SRR_approach
                    filename += "_itermax" + str(SRR_iter_max)
                    filename += "_alpha" + str(alpha)
                    HR_volume_alpha.write(directory=self._dir_results, filename=filename, write_mask=False)

                ## Compute prior term \f$ Psi(x) \f$
                self._Psis[i_reg_type, i] = self._compute_prior_term[SRR_approach](HR_volume_alpha)

                ## Compute residual \f$ \sum_k \Vert M_k y_k - M_k A_k x \Vert_{\ell^2}^2 \f$
                self._residuals[i_reg_type, i] = self._compute_residual(HR_volume_alpha)

        self.analyse(save_flag)


    def analyse(self, save_flag=0):   

        plot_colours = ["rx" , "bo"]

        plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')


        ## Plot
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(1,1,1)
        for i_reg_type in range(0, len(self._regularization_types)):

            ## Print on screen
            print("\t --- %s Regularization ---" %(self._regularization_types[i_reg_type]))
            print("\t\talpha\t\tPrior term Psi \t\tResidual")
            for i in range(0, len(self._alphas)):
                print("\t\t%s\t\t%.3e\t\t%.3e" %(self._alphas[i], self._Psis[i_reg_type, i], self._residuals[i_reg_type, i]))

            ax.plot(self._residuals[i_reg_type,:], self._Psis[i_reg_type,:], plot_colours[i_reg_type], label=self._regularization_types[i_reg_type])
            # ax.loglog(self._residuals[i_reg_type,:], self._Psis[i_reg_type,:], plot_colours[i_reg_type], label=self._regularization_types[i_reg_type])

        ## Show grid
        ax.grid()

        ## Add legend
        legend = ax.legend(loc="upper right")

        plt.title("L-curve\n$\Phi(\\vec{x}) = \\frac{1}{2}\sum_{k=1}^K \Vert M_k (\\vec{y}_k - A_k \\vec{x}) \Vert_{\ell^2}^2 + \\alpha \Psi(\\vec{x})\\rightarrow \min$" )
        # plt.title("L-curve for " + self._regularization_types[i_reg_type] + "Regularization")
        plt.ylabel("Prior term $\displaystyle\Psi(\\vec{x}_\\alpha)$")
        plt.xlabel("Residual $\sum_{k=1}^K \Vert M_k (\\vec{y}_k - A_k \\vec{x}_\\alpha) \Vert_{\ell^2}^2$")

        plt.draw()
        plt.pause(0.5) ## important! otherwise fig is not shown. Also needs plt.show() at the end of the file to keep figure open
        if save_flag:
            fig.savefig(self._dir_results + "L-curve.eps")

        
        # fig = plt.figure(2)
        # fig.clf()
        # ax = fig.add_subplot(1,1,1)
        # for i_reg_type in range(0, len(self._regularization_types)):

        #     ## Print on screen
        #     print("\t --- %s Regularization ---" %(self._regularization_types[i_reg_type]))
        #     print("\t\talpha\t\tPrior term Psi \t\tResidual")
        #     for i in range(0, len(self._alphas)):
        #         print("\t\t%s\t\t%.3e\t\t%.3e" %(self._alphas[i], self._Psis[i_reg_type, i], self._residuals[i_reg_type, i]))

        #     ax.loglog(self._residuals[i_reg_type,:], self._Psis[i_reg_type,:], plot_colours[i_reg_type], label=self._regularization_types[i_reg_type])

        # ## Show grid
        # ax.grid()

        # ## Add legend
        # legend = ax.legend(loc="upper right")

        # plt.title("L-curve")
        # # plt.title("L-curve for " + self._regularization_types[i_reg_type] + "Regularization")
        # plt.ylabel("Prior term Psi")
        # plt.xlabel("Residual")

        # plt.draw()
        # plt.pause(0.5) ## important! otherwise fig is not shown. Also needs plt.show() at the end of the file to keep figure open


    def _compute_prior_term_TK0(self, HR_volume):
        ## Get data array
        nda = self._itk2np.GetArrayFromImage( HR_volume.itk )

        return np.linalg.norm(nda)**2


    def _compute_prior_term_TK1(self, HR_volume):
        ## Get data array
        nda = self._itk2np.GetArrayFromImage( HR_volume.itk )

        ## Get kernels for differentiation
        kernel_Dx = self._get_forward_diff_x_kernel()
        kernel_Dy = self._get_forward_diff_y_kernel()
        kernel_Dz = self._get_forward_diff_z_kernel()

        ## Differentiate
        Dx = self._convolve(nda, kernel_Dx)
        Dy = self._convolve(nda, kernel_Dy)
        Dz = self._convolve(nda, kernel_Dz)

        ## Compute norm || Dx ||^2 with D = [D_x; D_y; D_z]
        return np.linalg.norm(Dx)**2 + np.linalg.norm(Dy)**2 + np.linalg.norm(Dz)**2


    ## Compute residual
    # \f[
    #   \sum_k \Vert M_k y_k - M_k A_k x \Vert_{\ell^2}^2
    # \f]
    # \param[in] HR_volume
    def _compute_residual(self, HR_volume):

        residual = 0.;

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            for j in range(0, stack.get_number_of_slices()):

                slice = slices[j]
                # sitkh.show_itk_image(slice.itk,title="slice")

                ## Add contribution || M_k (y_k - A_k x) ||^2
                residual += self._get_squared_norm_M_y_minus_Ax(slice, HR_volume)

        return residual


    def _get_squared_norm_M_y_minus_Ax(self, slice, HR_volume):

        ## TODO: Write itkBinaryFunctorImageFilter to multiply quicker
        ## Also, by changing to 
        ## itk.MultiplyImageFilter[itk.Image[itk.UC,3], image_type, image_type].New()
        ## the mask could be considered a different type, see sitkh.convert_sitk_to_itk_image
        ## (which in turn could help mask-based registration with ITK objects)
        # multiplier = itk.MultiplyImageFilter[itk.Image[itk.UC,3], image_type, image_type].New()
        multiplier = itk.MultiplyImageFilter[image_type, image_type, image_type].New()

        ## Update A_k based on relative position of slice
        self._update_oriented_gaussian_image_filter(slice, HR_volume)

        ## Compute A_k x
        Ax = self._A(HR_volume.itk)

        ## Compute y_k - A_k x
        y_minus_Ax = self._add_amplified_image(slice.itk, -1, Ax)

        ## Compute M_k (y_k - A_k x)
        multiplier.SetInput1( slice.itk_mask )
        multiplier.SetInput2( y_minus_Ax )
        multiplier.Update()

        M_y_minus_Ax = multiplier.GetOutput()
        M_y_minus_Ax.DisconnectPipeline()

        ## Compute || M_k (y_k - A_k x) ||^2
        nda = self._itk2np.GetArrayFromImage(M_y_minus_Ax)

        return np.linalg.norm(nda)**2


    ## Update internal Oriented Gaussian Interpolate Image Filter parameters. 
    #  Hence, update combined Downsample and Blur Operator according to the 
    #  relative position between slice and HR volume.
    #  \param[in] slice Slice object
    #  \param[in] HR_volume Stack object
    def _update_oriented_gaussian_image_filter(self, slice, HR_volume):
        ## Get variance covariance matrix representing Gaussian blurring in HR volume coordinates
        Cov_HR_coord = self._psf.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates( slice, HR_volume )

        ## Update parameters of forward operator A
        self._filter_oriented_Gaussian_interpolator.SetCovariance( Cov_HR_coord.flatten() )
        self._filter_oriented_Gaussian.SetOutputParametersFromImage( slice.itk )
        

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


    ## Compute convolution of array based on given kernel via 
    #  scipy.ndimage.convolve with "wrap" boundary conditions.
    #  \param[in] nda data array
    #  \param[in] kernel 
    #  \return data array convolved by given kernel
    def _convolve(self, nda, kernel):
        return ndimage.convolve(nda, kernel, mode='wrap')