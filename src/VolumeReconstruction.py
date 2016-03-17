## \file VolumeReconstruction.py
#  \brief Reconstruct volume given the current position of slices. 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015
# 
#  \version Reconstruction of HR volume by using Shepard's like method, Dec 2015
#  \version Reconstruction using Tikhonov regularization, Mar 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
import time                     


## Import modules from src-folder
import SimpleITKHelper as sitkh
import InverseProblemSolver as ips


## Class implementing the volume reconstruction given the current position of slices
class VolumeReconstruction:

    ## Constructor
    #  \param[in] stack_manager instance of StackManager containing all stacks and additional information
    #  \param[in] HR_volume Stack object containing the current estimate of the HR volume
    def __init__(self, stack_manager, HR_volume):

        ## Initialize variables
        self._stack_manager = stack_manager
        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()
        self._HR_volume = HR_volume

        ## Define dictionary to choose computational approach for reconstruction
        self._run_reconstruction = {
            "Shepard"           :   self._run_discrete_shepard_reconstruction,
            "Shepard-Deriche"   :   self._run_discrete_shepard_based_on_Deriche_reconstruction,
            "Tikhonov"          :   self._run_tikhonov_reconstruction
        }
        self._recon_approach = "Tikhonov"  # default reconstruction approach

        ## 1) Tikhonov reconstruction settings:
        self._solver = ips.InverseProblemSolver(self._stacks, self._HR_volume)

        ## Cut-off distance for Gaussian blurring filter
        self._alpha_cut = 3 

        ## Settings for optimizer
        self._alpha = 0.1               # Regularization parameter
        self._iter_max = 20             # Maximum iteration steps
        self._reg_type = 'TK0'          # Either Tikhonov zero- or first-order
        self._DTD_comp_type = "Laplace" #default value



    ## Get current estimate of HR volume
    #  \return current estimate of HR volume, instance of Stack
    def get_HR_volume(self):
        return self._HR_volume


    ## Set approach for reconstructing the HR volume. It can be either 
    #  'Tikhonov', 'Shepard' or 'Shepard-Deriche'
    #  \param[in] recon_approach either 'Tikhonov', 'Shepard' or 'Shepard-Deriche', string
    def set_reconstruction_approach(self, recon_approach):
        if recon_approach not in ["Tikhonov", "Shepard", "Shepard-Deriche"]:
            raise ValueError("Error: regularization type can only be either 'Tikhonov', 'Shepard' or 'Shepard-Deriche'")

        self._recon_approach = recon_approach


    ## Get chosen type of regularization.
    #  \return regularization type as string
    def get_reconstruction_approach(self):
        return self._recon_approach


    ## Computed reconstructed volume based on current estimated positions of slices
    def estimate_HR_volume(self):
        print("Estimate HR volume")

        t0 = time.clock()

        self._run_reconstruction[self._recon_approach]()

        time_elapsed = time.clock() - t0
        print("Elapsed time for SDA: %s seconds" %(time_elapsed))

        self._HR_volume.show()


    """
    Tikhonov regularization approach
    """
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


    ## Estimate the HR volume via Tikhonov regularization
    def _run_tikhonov_reconstruction(self):

        ## Set regularization parameter and maximum number of iterations
        self._solver.set_alpha( self._alpha )
        self._solver.set_iter_max( self._iter_max )
        self._solver.set_regularization_type( self._reg_type )
        self._solver.set_DTD_computation_type( self._DTD_comp_type)

        ## Perform reconstruction
        print("\n--- Run Tikhonov reconstruction algorithm ---")
        self._solver.run_reconstruction()


    """
    Shepard's like reconstruction approaches
    """
    ## Recontruct volume based on discrete Shepard's like method, cf. Vercauteren2006, equation (19).
    #  The computation here is based on the YVV variant of Recursive Gaussian Filter and executed
    #  via ITK
    #  \remark Obtained intensity values are positive.
    def _run_discrete_shepard_reconstruction(self):
        sigma = 0.5

        shape = sitk.GetArrayFromImage(self._HR_volume.sitk).shape
        helper_N_nda = np.zeros(shape)
        helper_D_nda = np.zeros(shape)

        default_pixel_value = 0.0

        for i in range(0, self._N_stacks):
        # for i in range(0, 2):
            print("  Stack %s/%s" %(i,self._N_stacks-1))
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()
            
            for j in range(0, N_slices):
                slice = slices[j]

                ## Nearest neighbour resampling of slice to target space (HR volume)
                slice_resampled_sitk = sitk.Resample(
                    slice.sitk, 
                    self._HR_volume.sitk, 
                    sitk.Euler3DTransform(), 
                    sitk.sitkNearestNeighbor, 
                    default_pixel_value,
                    self._HR_volume.sitk.GetPixelIDValue())

                ## Extract array of pixel intensities
                nda_slice = sitk.GetArrayFromImage(slice_resampled_sitk)

                ## Look for indices which are stroke by the slice in the isotropic grid
                ind_nonzero = nda_slice>0

                ## update arrays of numerator and denominator
                helper_N_nda[ind_nonzero] += nda_slice[ind_nonzero]
                helper_D_nda[ind_nonzero] += 1
                
                # print("helper_N_nda: (min, max) = (%s, %s)" %(np.min(helper_N_nda), np.max(helper_N_nda)))
                # print("helper_D_nda: (min, max) = (%s, %s)" %(np.min(helper_D_nda), np.max(helper_D_nda)))


        ## Create itk-images with correct header data
        # t0 = time.clock()
        pixel_type = itk.D
        dimension = 3
        image_type = itk.Image[pixel_type, dimension]
        # t1a = time.clock() - t0

        itk2np = itk.PyBuffer[image_type]
        # t1 = time.clock() - t0

        helper_N = itk2np.GetImageFromArray(helper_N_nda) 
        helper_D = itk2np.GetImageFromArray(helper_D_nda) 
        # t2 = time.clock() - t0

        helper_N.SetSpacing(self._HR_volume.sitk.GetSpacing())
        helper_N.SetDirection(sitkh.get_itk_direction_from_sitk_image(self._HR_volume.sitk))
        helper_N.SetOrigin(self._HR_volume.sitk.GetOrigin())

        helper_D.SetSpacing(self._HR_volume.sitk.GetSpacing())
        helper_D.SetDirection(sitkh.get_itk_direction_from_sitk_image(self._HR_volume.sitk))
        helper_D.SetOrigin(self._HR_volume.sitk.GetOrigin())
        # t3 = time.clock() - t0

        ## Apply Recursive Gaussian YVV filter
        gaussian = itk.SmoothingRecursiveYvvGaussianImageFilter[image_type, image_type].New()   # YVV-based Filter
        # gaussian = itk.SmoothingRecursiveGaussianImageFilter[image_type, image_type].New()    # Deriche-based Filter
        gaussian.SetSigma(sigma)
        gaussian.SetInput(helper_N)
        gaussian.Update()
        HR_volume_update_N = gaussian.GetOutput()
        # t4 = time.clock() - t0

        gaussian = itk.SmoothingRecursiveYvvGaussianImageFilter[image_type, image_type].New()   # YVV-based Filter
        # gaussian = itk.SmoothingRecursiveGaussianImageFilter[image_type, image_type].New()    # Deriche-based Filter
        gaussian.SetSigma(sigma)
        gaussian.SetInput(helper_D)
        gaussian.Update()
        HR_volume_update_D = gaussian.GetOutput()
        # t5 = time.clock() - t0

        ## Convert numerator and denominator back to data array
        nda_N = itk2np.GetArrayFromImage(HR_volume_update_N)
        nda_D = itk2np.GetArrayFromImage(HR_volume_update_D)
        # t6 = time.clock() - t0


        ## Compute data array of HR volume:
        # nda_D[nda_D==0]=1 
        nda = nda_N/nda_D.astype(float)
        # HR_volume_update.CopyInformation(self._HR_volume.sitk)
        # t7 = time.clock() - t0


        ## Update HR volume image file within Stack-object HR_volume
        HR_volume_update = sitk.GetImageFromArray(nda)
        HR_volume_update.CopyInformation(self._HR_volume.sitk)
        # t8 = time.clock() - t0

        ## Link HR_volume.sitk to the updated volume
        self._HR_volume.sitk = HR_volume_update


        # print("Elapsed time by image_type: %s seconds" %(t1a))
        # print("Elapsed time by initializing PyBuffer: %s seconds" %(t1))
        # print("Elapsed time by generating images from data arrays: %s seconds" %(t2))
        # print("Elapsed time by updating image headers: %s seconds" %(t3))
        # print("Elapsed time by smoothing numerator (YVV filter): %s seconds" %(t4))
        # print("Elapsed time by smoothing denominator (YVV filter): %s seconds" %(t5))
        # print("Elapsed time by fetching numerator and denominator data arrays: %s seconds" %(t6))
        # print("Elapsed time by division N/D: %s seconds" %(t7))
        # print("Elapsed time overall (after generating approximated HR volume): %s seconds" %(t8))



    ## Recontruct volume based on discrete Shepard's like method, cf. Vercauteren2006, equation (19).
    #  The computation here is based on the Deriche variant of Recursive Gaussian Filter and executed
    #  via SimpleITK. 
    #  \remark Obtained intensity values can be negative.
    def _run_discrete_shepard_based_on_Deriche_reconstruction(self):
        sigma = 0.5

        shape = sitk.GetArrayFromImage(self._HR_volume.sitk).shape
        helper_N_nda = np.zeros(shape)
        helper_D_nda = np.zeros(shape)

        default_pixel_value = 0.0

        for i in range(0, self._N_stacks):
        # for i in range(0, 2):
            print("  Stack %s/%s" %(i,self._N_stacks-1))
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()
            
            for j in range(0, N_slices):
                slice = slices[j]

                ## Nearest neighbour resampling of slice to target space (HR volume)
                slice_resampled_sitk = sitk.Resample(
                    slice.sitk, 
                    self._HR_volume.sitk, 
                    sitk.Euler3DTransform(), 
                    sitk.sitkNearestNeighbor, 
                    default_pixel_value,
                    self._HR_volume.sitk.GetPixelIDValue())

                ## Extract array of pixel intensities
                nda_slice = sitk.GetArrayFromImage(slice_resampled_sitk)

                ## Look for indices which are stroke by the slice in the isotropic grid
                ind_nonzero = nda_slice>0

                ## update arrays of numerator and denominator
                helper_N_nda[ind_nonzero] += nda_slice[ind_nonzero]
                helper_D_nda[ind_nonzero] += 1
                
                # print("helper_N_nda: (min, max) = (%s, %s)" %(np.min(helper_N_nda), np.max(helper_N_nda)))
                # print("helper_D_nda: (min, max) = (%s, %s)" %(np.min(helper_D_nda), np.max(helper_D_nda)))


        ## Create sitk-images with correct header data
        helper_N = sitk.GetImageFromArray(helper_N_nda) 
        helper_D = sitk.GetImageFromArray(helper_D_nda) 

        helper_N.CopyInformation(self._HR_volume.sitk)
        helper_D.CopyInformation(self._HR_volume.sitk)

        ## Apply recursive Gaussian smoothing
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)

        HR_volume_update_N = gaussian.Execute(helper_N)
        HR_volume_update_D = gaussian.Execute(helper_D)

        ## Avoid undefined division by zero
        """
        HACK start
        """
        ## HACK for denominator
        nda = sitk.GetArrayFromImage(HR_volume_update_D)
        ind_min = np.unravel_index(np.argmin(nda), nda.shape)
        # print nda[nda<0]
        # print nda[ind_min]

        eps = 1e-8
        # nda[nda<=eps]=1
        print("denominator min = %s" % np.min(nda))


        HR_volume_update_D = sitk.GetImageFromArray(nda)
        HR_volume_update_D.CopyInformation(self._HR_volume.sitk)

        ## HACK for numerator given that some intensities are negative!?
        nda = sitk.GetArrayFromImage(HR_volume_update_N)
        ind_min = np.unravel_index(np.argmin(nda), nda.shape)
        # nda[nda<=eps]=0
        # print nda[nda<0]
        print("numerator min = %s" % np.min(nda))
        """
        HACK end
        """
        
        ## Compute HR volume based on scattered data approximation with correct header (might be redundant):
        HR_volume_update = HR_volume_update_N/HR_volume_update_D
        HR_volume_update.CopyInformation(self._HR_volume.sitk)

        ## Update HR volume image file within Stack-object HR_volume
        self._HR_volume.sitk = HR_volume_update


        """
        Additional info
        """
        nda = sitk.GetArrayFromImage(HR_volume_update)
        print("Minimum of data array = %s" % np.min(nda))

