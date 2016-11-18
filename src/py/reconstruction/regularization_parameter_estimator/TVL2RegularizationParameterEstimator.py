#------------------------------------------------------------------------------
# \file TVL2RegularizationParameterEstimator.py
# \brief      This class serves to estimate the regularization parameter for
#             Tv-L2 regularization computed via ADMM algorithm.
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
sys.path.append(dir_src_root)

## Import modules
from reconstruction.regularization_parameter_estimator.RegularizationParameterEstimator import RegularizationParameterEstimator
import reconstruction.solver.TVL2Solver as tvl2
import base.Stack as st


class TVL2RegularizationParameterEstimator(RegularizationParameterEstimator):
    
    ##
    #       Constructor
    # \date       2016-08-01 23:40:34+0100
    #
    # \param      self                                    The object
    # \param[in]  stacks                                  list of Stack objects
    #                                                     containing all stacks
    #                                                     used for the
    #                                                     reconstruction
    # \param[in]  HR_volume                               Stack object
    #                                                     containing the
    #                                                     current estimate of
    #                                                     the HR volume (used
    #                                                     as initial value +
    #                                                     space definition)
    # \param[in]  alpha_cut                               Cut-off distance for
    #                                                     Gaussian blurring
    #                                                     filter
    # \param[in]  iter_max                                number of maximum
    #                                                     iterations, scalar
    # \param[in]  alpha_array                             array containing
    #                                                     regularization
    #                                                     parameter to sweep
    #                                                     through, list
    # \param[in]  dir_results                             Directory to store
    #                                                     computed results. If
    #                                                     'None' no results are
    #                                                     written.
    # \param[in]  filename_results_prefix                 Prefix applied for
    #                                                     each filename written
    #                                                     to dir_results
    # \param[in]  ADMM_iterations                         number of ADMM
    #                                                     iterations, scalar
    # \param[in]  rho_array                               Array containing
    #                                                     regularization
    #                                                     parameter of
    #                                                     augmented Lagrangian
    #                                                     term, list
    # \param[in]  ADMM_iterations_output_dir              The ADMM iterations
    #                                                     output dir
    # \param[in]  ADMM_iterations_output_filename_prefix  The ADMM iterations
    #                                                     output filename
    #                                                     prefix
    #
    def __init__(self, stacks, HR_volume, alpha_cut=3, iter_max=10, alpha_array=[None], dir_results="RegularizationParameterEstimation/", filename_results_prefix="", ADMM_iterations=5, rho_array=[None], ADMM_iterations_output_dir="TV-L2_ADMM_iterations/", ADMM_iterations_output_filename_prefix="TV-L2"):

        ## Run constructor of superclass
        RegularizationParameterEstimator.__init__(self, stacks, HR_volume, alpha_cut=alpha_cut, iter_max=iter_max, alpha_array=alpha_array, dir_results=dir_results, filename_results_prefix=filename_results_prefix)               
        
        self._rho_array = rho_array
        self._ADMM_iterations = ADMM_iterations   # Number of performed ADMM iterations
        self._ADMM_iterations_output_dir = self._dir_results + ADMM_iterations_output_dir
        self._ADMM_iterations_output_filename_prefix = ADMM_iterations_output_filename_prefix


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
    def run_reconstructions(self, save_flag=1, estimate_initial_value=False):

        ## Run method of superclass
        # RegularizationParameterEstimator._prepare_run_reconstructions()

        ## Create output directory in case it is not existing
        os.system("mkdir -p " + self._dir_results)

        ## Get total number of alphas to test
        N_rhos = len(self._rho_array)
        N_alphas = len(self._alpha_array)

        ## Create file carrying L-curve information
        self._filename_of_txt_file, header = self._get_filename_and_header()
        self._create_file(self._filename_of_txt_file, header)

        ## Iterate over all given rhos
        for i_rho in range(0, N_rhos):
            rho = self._rho_array[i_rho]

            ## Iterate over all given alphas
            for i_alpha in range(0, N_alphas):
                alpha = self._alpha_array[i_alpha]

                print("\n\t--- TV-L2-Regularization: Reconstruction with rho = %s (%s,%s) and alpha = %s (%s/%s) ---" %(rho, i_rho+1, N_rhos, alpha, i_alpha+1, N_alphas))

                ## Initialize solver
                HR_volume_init = st.Stack.from_stack(self._HR_volume)
                solver = tvl2.TVL2Solver(self._stacks, HR_volume_init, alpha_cut=self._alpha_cut, alpha=alpha, iter_max=self._iter_max, rho=rho, ADMM_iterations=self._ADMM_iterations, ADMM_iterations_output_dir=self._ADMM_iterations_output_dir, ADMM_iterations_output_filename_prefix=self._ADMM_iterations_output_filename_prefix)

                ## Reconstruct volume for given alpha
                solver.run_reconstruction(estimate_initial_value=estimate_initial_value)
                HR_volume_alpha = solver.get_HR_volume()

                ## Compute reconstruction statistics
                solver.compute_statistics()
                solver.print_statistics()

                ## Write reconstructed nifti-volume
                if save_flag:
                    filename_image = self._get_filename_reconstructed_image(alpha, rho)
                    HR_volume_alpha.write(directory=self._dir_results, filename=filename_image, write_mask=False)

                ## Write L-curve information into file
                residual_data_fit = solver.get_residual_ell2()
                residual_prior = solver.get_residual_prior()
                computational_time = solver.get_computational_time()

                array_out = np.array([i_rho+1, i_alpha+1, rho, alpha, residual_data_fit, residual_prior]).reshape(1,-1)
                format = "%d %d %.3f %.3f %.10e %.10e"
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
        filename += "TV-L2-regularization"
        # filename += "_alpha" + str(alpha)
        # filename += "_rho" + str(rho)
        filename += "_ADMMiterations" + str(self._ADMM_iterations)
        filename += "_TK1itermax" + str(self._iter_max)
        filename += "_" + str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2)
        filename += "_" + str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

        header = "## " + "TV-L2-Regularization"
        header += " with " + str(self._ADMM_iterations) + " ADMM iterations."
        header += " TK1-solver settings:" 
        header += " itermax = " + str(self._iter_max)
        # header += ", tolerance = " + str(SRR_tolerance)
        header += " (" + str(now.day).zfill(2) + "." + str(now.month).zfill(2) + "." + str(now.year) 
        header += ", " + str(now.hour).zfill(2) + ":" + str(now.minute).zfill(2) + ":" + str(now.second).zfill(2) + ")"
        header += "\n## " + "rho" + "\t" + "alpha" + "\t" + "Residual data fit"+ "\t" + "Psi"
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
    def _get_filename_reconstructed_image(self, alpha, rho):

        ## Only use signifcant digits for string
        alpha_str = "%g" % alpha
        rho_str = "%g" % rho

        ## Build filename
        filename_image =  self._filename_results_prefix
        filename_image += "TVL2"
        filename_image += "_stacks" + str(self._N_stacks)
        filename_image += "_rho" + rho_str
        filename_image += "_alpha" + alpha_str
        filename_image += "_TK1itermax" + str(self._iter_max)
        filename_image += "_ADMMiterations" + str(self._ADMM_iterations)

        ## Replace decimal point by 'p'
        filename_image = filename_image.replace(".", "p")

        return filename_image


        