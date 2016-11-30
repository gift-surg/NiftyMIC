#------------------------------------------------------------------------------
# \file TikhonovRegularizationParameterEstimator.py
# \brief      This class serves to estimate the regularization parameter for
#             Tikhonov regularization
# 
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2016
# 


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np
import datetime
import re               #regular expression

## Add directories to import modules
dir_src_root = "../../src/py/"
sys.path.append(dir_src_root )

## Import modules
from reconstruction.regularization_parameter_estimator.RegularizationParameterEstimator import RegularizationParameterEstimator
import reconstruction.solver.TikhonovSolver as tk
import base.Stack as st


class TikhonovRegularizationParameterEstimator(RegularizationParameterEstimator):
    
    ##
    #       Constructor
    # \date       2016-08-01 23:40:34+0100
    #
    # \param      self                     The object
    # \param[in]  stacks                   list of Stack objects containing all
    #                                      stacks used for the reconstruction
    # \param[in]  HR_volume                Stack object containing the current
    #                                      estimate of the HR volume (used as
    #                                      initial value + space definition)
    # \param[in]  alpha_cut                Cut-off distance for Gaussian
    #                                      blurring filter
    # \param[in]  iter_max                 number of maximum iterations, scalar
    # \param[in]  alpha_array              array containing regularization
    #                                      parameter to sweep through, list
    # \param[in]  dir_results              Directory to store computed results.
    #                                      If 'None' no results are written.
    # \param[in]  filename_results_prefix  Prefix applied for each filename
    #                                      written to dir_results
    # \param[in]  reg_type                 Type of regularization applied.
    #                                      Either zeroth-order ('TK0') or
    #                                      first-order ('TK1') Tikhonov
    #
    def __init__(self, stacks, HR_volume, alpha_cut=3, iter_max=10, alpha_array=[None], dir_results="RegularizationParameterEstimation/", filename_results_prefix="", reg_type="TK1", minimizer="lsmr", deconvolution_mode="full_3D", predefined_covariance=None):

        ## Run constructor of superclass
        RegularizationParameterEstimator.__init__(self, stacks, HR_volume, alpha_cut=alpha_cut, iter_max=iter_max, alpha_array=alpha_array, dir_results=dir_results, filename_results_prefix=filename_results_prefix)               
        ## Set regularization type
        self._reg_type = reg_type

        self._minimizer = minimizer
        self._deconvolution_mode = deconvolution_mode
        self._predefined_covariance = predefined_covariance


    ##
    #       Sets the regularization type.
    # \date       2016-08-01 19:53:16+0100
    #
    # \param      self      The object
    # \param      reg_type  Registration type. Either 'TK0' or 'TK1'
    #
    def set_regularization_type(self, reg_type):
        self._reg_type = reg_type


    ##
    #       Run reconstruction for several alphas based on Tikhonov
    #             regularization
    # \date       2016-08-01 19:47:53+0100
    # \post       "filename" contains the name of the text-file containing the
    #             results which can be fetched via get_filename_of_txt_file
    #
    # \param      self       The object
    # \param      save_flag  Decide whether reconstructed images shall be
    #                        written to dir_results
    #
    def run_reconstructions(self, save_flag=1):

        ## Run method of superclass
        # RegularizationParameterEstimator._prepare_run_reconstructions()

        ## Create output directory in case it is not existing
        os.system("mkdir -p " + self._dir_results)

        ## Get total number of alphas to test
        N_alphas = len(self._alpha_array)

        ## Create file carrying L-curve information
        self._filename_of_txt_file, header = self._get_filename_and_header()
        self._create_file(self._filename_of_txt_file, header)

        ## Iterate over all given alphas
        for i_alpha in range(0, N_alphas):
            alpha = self._alpha_array[i_alpha]
            
            print("\n\t--- %s-Regularization: Reconstruction with alpha = %s (%s/%s) ---" %(self._reg_type, alpha, i_alpha+1, N_alphas))

            ## Initialize solver
            HR_volume_init = st.Stack.from_stack(self._HR_volume)
            solver = tk.TikhonovSolver(self._stacks, HR_volume_init, alpha_cut=self._alpha_cut, alpha=alpha, iter_max=self._iter_max, reg_type=self._reg_type, minimizer=self._minimizer, deconvolution_mode=self._deconvolution_mode, predefined_covariance=self._predefined_covariance)

            ## Reconstruct volume for given alpha
            solver.run_reconstruction()
            HR_volume_alpha = solver.get_HR_volume()

            ## Compute reconstruction statistics
            solver.compute_statistics()
            solver.print_statistics()

            ## Write reconstructed nifti-volume
            if save_flag:
                filename_image = self._get_filename_reconstructed_image(alpha)
                HR_volume_alpha.write(directory=self._dir_results, filename=filename_image, write_mask=False)

            ## Write L-curve information into file
            residual_data_fit = solver.get_residual_ell2()
            residual_prior = solver.get_residual_prior()
            computational_time = solver.get_computational_time()

            array_out = np.array([i_alpha+1, alpha, residual_data_fit, residual_prior]).reshape(1,-1)
            format = "%d %.3f %.10e %.10e"
            self._write_array_to_file(self._filename_of_txt_file, array_out, format=format, delimiter="\t")


    ##
    #       Gets the filename and header.
    # \date       2016-08-01 19:42:35+0100
    #
    # \param      self  The object
    #
    # \return     The filename and header.
    #
    def _get_filename_and_header(self):

        now = datetime.datetime.now()

        filename = self._filename_results_prefix
        filename += self._reg_type
        filename += "-regularization"
        filename += "_" + self._minimizer
        filename += "_itermax" + str(self._iter_max)
        filename += "_" + str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2)
        filename += "_" + str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

        header = "## " + self._reg_type + "-Regularization"
        header += ", itermax = " + str(self._iter_max)
        # header += ", tolerance = " + str(SRR_tolerance)
        header += " (" + str(now.day).zfill(2) + "." + str(now.month).zfill(2) + "." + str(now.year) 
        header += ", " + str(now.hour).zfill(2) + ":" + str(now.minute).zfill(2) + ":" + str(now.second).zfill(2) + ")"
        header += "\n## " + "\t" + "alpha" + "\t" + "Residual data fit"+ "\t" + "Psi"
        header += "\n"

        return filename, header


    ##
    #       Gets the filename of reconstructed image.
    # \date       2016-08-01 19:42:46+0100
    #
    # \param      self   The object
    # \param      alpha  Regularization parameter
    #
    # \return     The filename reconstructed image.
    #
    def _get_filename_reconstructed_image(self, alpha):

        ## Only use signifcant digits for string
        alpha_str = "%g" % alpha

        ## Build filename
        filename_image =  self._filename_results_prefix
        filename_image += self._reg_type
        filename_image += "_stacks" + str(self._N_stacks)
        filename_image += "_" + self._minimizer
        filename_image += "_alpha" + alpha_str
        filename_image += "_itermax" + str(self._iter_max)

        ## Replace decimal point by 'p'
        filename_image = filename_image.replace(".", "p")

        return filename_image


        