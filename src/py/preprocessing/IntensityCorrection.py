##
# \file IntensityCorrection.py
# \brief      Class containing functions to correct for intensities
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
#


## Import libraries
import sys
import SimpleITK as sitk
import numpy as np
from scipy.optimize import least_squares
import time

## Import modules
import base.Stack as st
import utilities.SimpleITKHelper as sitkh
import utilities.PythonHelper as ph
import matplotlib.pyplot as plt


##
#       Class to correct intensities
# \date       2016-11-01 20:12:46+0000
#
class IntensityCorrection(object):

    ##
    #       Constructor
    # \date       2016-11-01 21:57:13+0000
    #
    # \param      self                             The object
    # \param      stack                            Stack object to be intensity
    #                                              corrected
    # \param      reference                        Stack object used as
    #                                              reference for intensities
    #                                              (needs to be in the physical
    #                                              space as stack)
    # \param      use_reference_mask               Use reference mask (as given
    #                                              in \p reference) to reduce
    #                                              focus for intensity
    #                                              correction; bool
    # \param      use_individual_slice_correction  State whether intensity
    #                                              correction is performed for
    #                                              each slice independently;
    #                                              bool
    # \param      use_verbose                      Verbose; bool
    #
    def __init__(self, stack=None, reference=None, use_reference_mask=True, use_individual_slice_correction=False, use_verbose=False, additional_stack=None):

        if stack is not None:
            self._stack = st.Stack.from_stack(stack)
        else:
            self._stack = None

        ## Additional stack to correct alongside given stack
        if additional_stack is not None:
            self._additional_stack = st.Stack.from_stack(additional_stack)
        else:
            self._additional_stack = None

        ## Check that stack and reference are in the same space
        if reference is not None:
            try:
                self._stack.sitk - reference.sitk
            except:
                raise ValueError("Reference and stack are not in the same space")
            self._reference = st.Stack.from_stack(reference)
        else:
            self._reference = None

        self._apply_intensity_correction = {
            "linear"    : self._apply_linear_intensity_correction,
            "affine"    : self._apply_affine_intensity_correction,
        }

        self._use_verbose = use_verbose
        self._use_reference_mask = use_reference_mask
        self._use_individual_slice_correction = use_individual_slice_correction


    ##
    #       Sets the stack.
    # \date       2016-11-05 22:58:01+0000
    #
    # \param      self   The object
    # \param      stack  The stack as Stack object
    #
    def set_stack(self, stack):
        self._stack = st.Stack.from_stack(stack)


    ##
    #       Sets the reference.
    # \date       2016-11-05 22:58:10+0000
    #
    # \param      self       The object
    # \param      reference  The reference as Stack object
    #
    def set_reference(self, reference):
        self._reference = st.Stack.from_stack(reference)


    ##
    # Sets additional stack to correct alongside given stack
    # \date       2016-12-05 12:19:25+0000
    #
    # \param      self              The object
    # \param      additional_stack  The additional stack
    #
    # \return     { description_of_the_return_value }
    #
    def set_additional_stack(self, additional_stack):
        self._additional_stack = st.Stack.from_stack(additional_stack)


    ##
    #       Use verbose
    # \date       2016-11-05 22:58:28+0000
    #
    # \param      self     The object
    # \param      verbose  The verbose as boolean
    #
    def use_verbose(self, verbose):
        self._use_verbose = verbose


    def use_reference_mask(self, use_reference_mask):
        self._use_reference_mask = use_reference_mask

    ##
    # Sets the use individual slice correction.
    # \date       2016-11-22 22:47:47+0000
    #
    # \param      self  The object
    # \param      flag  The flag
    #
    def use_individual_slice_correction(self, flag):
        self._use_individual_slice_correction = flag


    ##
    #       Gets the intensity corrected stack
    # \date       2016-11-05 22:58:50+0000
    #
    # \param      self  The object
    #
    # \return     The intensity corrected stack as Stack object
    #
    def get_intensity_corrected_stack(self):
        return st.Stack.from_stack(self._stack)


    def get_intensity_corrected_additional_stack(self):
        return st.Stack.from_stack(self._additional_stack)

    ##
    #       Gets the intensity correction coefficients obtained for each
    #             slice of the stack.
    # \date       2016-11-10 02:22:40+0000
    #
    # \param      self  The object
    #
    # \return     The intensity correction coefficients as (N_slices x
    #             DOF)-array
    #
    def get_intensity_correction_coefficients(self):
        return np.array(self._correction_coefficients)


    ##
    #       Clip lower intensities based on percentile threshold
    # \date       2016-11-05 22:59:08+0000
    #
    # \param      self        The object
    # \param      percentile  The percentile defining the threshold
    #
    def run_lower_percentile_capping_of_stack(self, percentile=10):

        print("Cap lower intensities at %d%%-percentile" %(percentile))

        nda = sitk.GetArrayFromImage(self._stack.sitk)

        ## Clip lower intensity values
        i0 = np.percentile(nda, percentile)
        nda[np.where(nda<i0)] = 0
        nda[np.where(nda>=i0)] -= i0

        ## Create Stack instance with correct image header information        
        self._stack = self._create_stack_from_corrected_intensity_array(nda, self._stack)

        if self._additional_stack is not None:
            nda_additional_stack = sitk.GetArrayFromImage(self._additional_stack.sitk)

            nda_additional_stack[np.where(nda_additional_stack<i0)] = 0
            nda_additional_stack[np.where(nda_additional_stack>=i0)] -= i0

            ## Create Stack instance with correct image header information        
            self._additional_stack = self._create_stack_from_corrected_intensity_array(nda_additional_stack, self._additional_stack)


    ##
    #       Run linear intensity correction model.
    # \date       2016-11-05 23:02:46+0000
    #
    # Perform linear intensity correction, i.e. 
    #    minimize || reference - c1*stack || in ell^2-sense.
    #
    # \param      self  The object
    #
    def run_linear_intensity_correction(self):
        self._stack, self._correction_coefficients, self._additional_stack = self._run_intensity_correction("linear")


    ##
    #       Run affine intensity correction model.
    # \date       2016-11-05 23:05:48+0000
    #
    # Perform affine intensity correction, i.e. 
    #    minimize || reference - (c1*stack + c0)|| in ell^2-sense.
    #    
    # \param      self  The object
    #
    def run_affine_intensity_correction(self):
        self._stack, self._correction_coefficients, self._additional_stack = self._run_intensity_correction("affine")


    ##
    #       Execute respective intensity correction model.
    # \date       2016-11-05 23:06:37+0000
    #
    # \param      self              The object
    # \param      correction_model  The correction model. Either 'linear' or
    #                               'affine'
    #
    def _run_intensity_correction(self, correction_model):

        N_slices = self._stack.get_number_of_slices()

        if correction_model in ["linear"]:
            correction_coefficients = np.zeros((N_slices, 1))
        elif correction_model in ["affine"]:
            correction_coefficients = np.zeros((N_slices, 2))

        ## Gets the required data arrays to perform intensity correction
        nda, nda_reference, nda_mask, nda_additional_stack = self._get_data_arrays_prior_to_intensity_correction()

        if self._use_individual_slice_correction:
            print("Run " + correction_model + " intensity correction for each slice individually")
            for i in range(0, N_slices):
                if self._use_verbose:
                    sys.stdout.write("Slice %2d/%d: " %(i, self._stack.get_number_of_slices()-1))
                    sys.stdout.flush()
                if self._additional_stack is None:
                    nda[i,:,:], correction_coefficients[i,:] = self._apply_intensity_correction[correction_model](nda[i,:,:], nda_reference[i,:,:], nda_mask[i,:,:])
                else:
                    nda[i,:,:], correction_coefficients[i,:], nda_additional_stack[i,:,:] = self._apply_intensity_correction[correction_model](nda[i,:,:], nda_reference[i,:,:], nda_mask[i,:,:], nda_additional_stack[i,:,:])
        else:
            print("Run " + correction_model + " intensity correction uniformly for entire stack")
            if self._additional_stack is None:
                nda, cc = self._apply_intensity_correction[correction_model](nda, nda_reference, nda_mask)
            else:
                nda, cc, nda_additional_stack = self._apply_intensity_correction[correction_model](nda, nda_reference, nda_mask, nda_additional_stack)
            correction_coefficients[:,] = np.tile(cc, (N_slices,1))
        
        ## Create Stack instance with correct image header information
        if self._additional_stack is None:
            return self._create_stack_from_corrected_intensity_array(nda, self._stack), correction_coefficients, None
        else:
            return self._create_stack_from_corrected_intensity_array(nda, self._stack), correction_coefficients, self._create_stack_from_corrected_intensity_array(nda_additional_stack, self._additional_stack)



    ##
    #       Perform affine intensity correction via normal equations
    # \date       2016-11-05 23:10:49+0000
    #
    # \param      self                  The object
    # \param      nda                   Data array to be corrected
    # \param      nda_mask              Mask to be used
    # \param      nda_reference  Masked reference data array used to
    #                                   compute coefficients
    #
    # \return     intensity corrected data array as np.array
    #
    def _apply_affine_intensity_correction(self, nda, nda_reference, nda_mask, nda_additional_stack=None):


        ## Find masked indices
        indices = np.where(nda_mask>0)

        ## Model: y = x*c1 + c0 = [x, 1]*[c1, c0]' = A*[c1,c0]
        x = nda[indices].flatten().astype('double')
        y = nda_reference[indices].flatten().astype('double')

        ## Solve via normal equations: [c1, c0] = (A'A)^{-1}A'y
        A = np.ones((x.size,2))
        A[:,0] = x

        B = np.linalg.pinv(A.transpose().dot(A)).dot(A.transpose())
        c1, c0 = B.dot(y)

        if self._use_verbose:
            print("(c1, c0) = (%.3f, %.3f)" %(c1, c0))

        if nda_additional_stack is None:
            return nda*c1 + c0, np.array([c1,c0])
        else:
            return nda*c1 + c0, np.array([c1,c0]), nda_additional_stack*c1 + c0


    ##
    #       Perform linear intensity correction via normal equations
    # \date       2016-11-05 23:12:13+0000
    #
    # \param      self                  The object
    # \param      nda                   Data array to be corrected
    # \param      nda_mask            Mask to be used
    # \param      nda_reference  Masked reference data array used to
    #                                   compute coefficients
    #
    # \return     intensity corrected data array as np.array
    #
    def _apply_linear_intensity_correction(self, nda, nda_reference, nda_mask, nda_additional_stack=None):

        ## Find masked indices
        indices = np.where(nda_mask>0)

        ## Model: y = x*c1
        x = nda.flatten().astype('double')
        y = nda_reference.flatten().astype('double')

        # ph.show_2D_array_list([nda, nda_reference])
        ## Solve via normal equations: c1 = x'y/(x'x)
        c1 = x.dot(y)/x.dot(x)

        if self._use_verbose:
            print("c1 = %.3f" %(c1))
        
        if nda_additional_stack is None:
            return nda*c1, c1
        else:
            return nda*c1, c1, nda_additional_stack*c1


    ##
    #       Gets the data arrays prior to intensity correction.
    # \date       2016-11-05 23:07:37+0000
    #
    # \param      self  The object
    #
    # \return     The data arrays prior to intensity correction.
    #
    def _get_data_arrays_prior_to_intensity_correction(self):
        
        ## Get required data arrays for intensity correction
        nda = sitk.GetArrayFromImage(self._stack.sitk)
        nda_reference = sitk.GetArrayFromImage(self._reference.sitk)

        if self._use_reference_mask:
            nda_mask = sitk.GetArrayFromImage(self._reference.sitk_mask)
        else:
            nda_mask = np.ones_like(nda)

        if self._additional_stack is None:
            nda_additional_stack = None
        else:
            nda_additional_stack = sitk.GetArrayFromImage(self._additional_stack.sitk)

        return nda, nda_reference, nda_mask, nda_additional_stack


    ##
    #       Creates a Stack object from corrected intensity array with
    #             same image header information as input \p stack.
    # \date       2016-11-05 23:15:33+0000
    #
    # \param      self  The object
    # \param      nda   The nda
    #
    # \return     Stack object with image containing the given array
    #             information.
    #
    def _create_stack_from_corrected_intensity_array(self, nda, stack):

        ## Convert back to image with correct header
        image_sitk = sitk.GetImageFromArray(nda)
        image_sitk.CopyInformation(stack.sitk)

        return st.Stack.from_sitk_image(image_sitk, stack.get_filename(), stack.sitk_mask)


