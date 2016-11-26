#------------------------------------------------------------------------------
# \file AnalysisRegularizationParameterEstimator.py
# \brief      This is the "main-file" to perform the analysis to estimate the
#             regularization parameters for (both Tikhonov and TV-L2)
#             regularization.
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2016
#

## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
import time  
from scipy.optimize import curve_fit
from scipy import ndimage
import matplotlib.pyplot as plt
import datetime
import re               #regular expression

## Add directories to import modules
dir_src_root = "../../src/py/"
sys.path.append(dir_src_root)

## Import modules
import utilities.SimpleITKHelper as sitkh

import reconstruction.solver.TikhonovSolver as tk
import reconstruction.regularization_parameter_estimator.TikhonovRegularizationParameterEstimator as tkrpe
import reconstruction.regularization_parameter_estimator.TVL2RegularizationParameterEstimator as tvl2rpe

class AnalysisRegularizationParameterEstimator(object):

    ##
    #       Constructor
    # \date       2016-08-01 23:40:34+0100
    #
    # \param      self                                    The object
    # \param[in]  stacks                                  list of Stack objects
    #                                                     containing all stacks
    #                                                     used for the
    #                                                     reconstruction
    # \param[in]  HR_volume_init                          Stack object
    #                                                     containing the
    #                                                     current estimate of
    #                                                     the HR volume (used
    #                                                     as initial value +
    #                                                     space definition)
    # \param[in]  dir_results                             Directory to store
    #                                                     computed results. If
    #                                                     'None' no results are
    #                                                     written.
    # \param[in]  alpha_cut                               Cut-off distance for
    #                                                     Gaussian blurring
    #                                                     filter
    # \param[in]  iter_max                                number of maximum
    #                                                     iterations, scalar
    # \param[in]  alpha_array                             array containing
    #                                                     regularization
    #                                                     parameter to sweep
    #                                                     through, list
    # \param[in]  rho_array                               Array containing
    #                                                     regularization
    #                                                     parameter of
    #                                                     augmented Lagrangian
    #                                                     term, list
    # \param[in]  ADMM_iterations                         number of ADMM
    #                                                     iterations, scalar
    # \param[in]  ADMM_iterations_output_dir              The ADMM iterations
    #                                                     output dir
    # \param[in]  ADMM_iterations_output_filename_prefix  The ADMM iterations
    #                                                     output filename
    #                                                     prefix
    #
    def __init__(self, stacks, HR_volume_init, dir_results="/tmp/AnalysisRegularizationParameterEstimation/", alpha_cut=3, iter_max=10, alpha_array=[0.01, 0.05, 0.1, 0.5], rho_array=[0.5, 1, 2], ADMM_iterations=5, ADMM_iterations_output_dir="TV-L2_ADMM_iterations/", ADMM_iterations_output_filename_prefix="TV-L2", minimizer="lsmr", deconvolution_mode="full_3D", predefined_covariance=None):

        self._stacks = stacks
        self._HR_volume_init = HR_volume_init

        self._dir_results = dir_results
        self._alpha_cut = alpha_cut
        self._iter_max = iter_max
        self._alpha_array = alpha_array
        self._ADMM_iterations = ADMM_iterations
        self._rho_array = rho_array
        self._ADMM_iterations_output_dir = ADMM_iterations_output_dir
        self._ADMM_iterations_output_filename_prefix = ADMM_iterations_output_filename_prefix

        self._minimizer = minimizer
        self._deconvolution_mode = deconvolution_mode
        self._predefined_covariance = predefined_covariance

        ## colors: r,b,g,c,m,y,k,w (http://matplotlib.org/api/colors_api.html)
        ## markers: http://matplotlib.org/api/markers_api.html#module-matplotlib.markers
        ## line styles: same as in Matlab (http://matplotlib.org/api/lines_api.html#matplotlib.lines.Line2D.set_linestyle)
        self._PLOT_FORMAT = ["rx:" , "bo:" , "gs:", "m<:", "c>:", "y^:", "kv:"]
        # self._PLOT_FORMAT = ["rx:" , "bo:" , "gs:", "r<-.", "b>-.", "g^-."]

        self._run_reconstructions = {
            "TK0"   : self._run_reconstructions_TK0,
            "TK1"   : self._run_reconstructions_TK1,
            "TV-L2" : self._run_reconstructions_TVL2
        }

        self._print_data_on_screen = {
            "TK0"   : self._print_data_on_screen_TK0,
            "TK1"   : self._print_data_on_screen_TK1,
            "TV-L2" : self._print_data_on_screen_TVL2
        }

        self._plot_add_curve = {
            "TK0"   : self._plot_add_curve_TK0,
            "TK1"   : self._plot_add_curve_TK1,
            "TV-L2" : self._plot_add_curve_TVL2
        }

        self._filenames = []
        self._reg_types = None



    ##
    #       Set the types of regularization used to sweep through the
    #             regularization parameters (alpha and rho where applicable)
    # \date       2016-08-02 00:08:50+0100
    #
    # \param      self       The object
    # \param      reg_types  Types of regularization, list. List can contain
    #                        'TK0', 'TK1' and 'TV-L2'.
    #
    def set_regularization_types(self, reg_types):
        self._reg_types = reg_types


    ##
    #       Set the directory to store the computed results or, in case
    #             results have already been computed, where existing results
    #             are located.
    # \date       2016-08-02 00:11:21+0100
    #
    # \param      self         The object
    # \param      dir_results  The directory to store the results, string
    #
    def set_directory_results(self, dir_results):
        self._dir_results = dir_results



    ##
    #       In case existing files shall be used to perform the L-curve
    #             analysis, you can specify those here. Hence, specify
    #             filenames and dir_results before running 'show_L_curves'.
    # \date       2016-08-02 00:12:01+0100
    #
    # \param      self       The object
    # \param      filenames  The filenames
    #
    def set_filenames(self, filenames):
        self._filenames = filenames


    ##
    #       Reconstruct volumes for each type of regularization by
    #             sweeping through the parameter space specified by alpha (and
    #             rho)
    # \date       2016-08-02 00:14:44+0100
    # \post       "filenames" contains the list of txt-files produced. They can
    #             be analysed directly by running "show_L_curves".
    #
    # \param      self  The object
    #
    def run_reconstructions(self):

        for i in range(0,len(self._reg_types)):
            reg_type = self._reg_types[i]

            self._run_reconstructions[reg_type]()


    ##
    #       Shows the L-curves based on the infomration stored in
    #             filenames and dir_results
    # \date       2016-08-02 00:17:03+0100
    #
    # \param      self         The object
    # \param      save_figure  Save figure as eps-file to dir_results
    #
    def show_L_curves(self, save_figure=False):

        ## Plot
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(1,1,1)

        ## Iterate through all computed/specified files
        filename_out = ""
        for i_file in range(0, len(self._filenames)):

            ## Detect which SRR approach was used for the results stored in this file
            SRR_approach = self._get_regularization_approach(self._filenames[i_file])

            # data = np.loadtxt(self._dir_results + self._filenames[i_file] + ".txt" , delimiter="\t", skiprows=2)
            data = np.loadtxt(self._dir_results + self._filenames[i_file] + ".txt" , skiprows=2)

            ## Print on screen
            self._print_data_on_screen[SRR_approach](data)

            ## Plot curve
            self._plot_add_curve[SRR_approach](data, i_file, ax)

            ## Prepare filename in case used
            filename_out += SRR_approach + "_"

        ## Add all figure information
        self._plot_add_figure_information(ax)

        if save_figure:
            now = datetime.datetime.now()
            filename_out += str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2)
            filename_out += "_" + str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

            fig.savefig(self._dir_results + filename_out + ".eps")


    ##
    #       Return the regularization approach used to compute results
    #             stored in filename
    # \date       2016-08-02 00:53:54+0100
    #
    # \param      self      The object
    # \param      filename  The filename, string
    #
    # \return     The regularization approach, string
    #
    def _get_regularization_approach(self, filename):
        if "TK0-regularization" in filename:
            return "TK0"
        elif "TK1-regularization" in filename:
            return "TK1"
        else:
            return "TV-L2"


    ##
    #       Reconstruct the volumes based on zeroth-order Tikhonov
    #             regularization by sweeping through all alphas specified in
    #             alpha_array
    # \date       2016-08-02 00:20:20+0100
    #
    # \param      self  The object
    #
    def _run_reconstructions_TK0(self):

        ## Initialize zeroth-order Tikhonov solver
        regularization_parameter_estimator = tkrpe.TikhonovRegularizationParameterEstimator(self._stacks, self._HR_volume_init, alpha_cut=self._alpha_cut, iter_max=self._iter_max, alpha_array=self._alpha_array, dir_results=self._dir_results, reg_type="TK0", minimizer=self._minimizer, deconvolution_mode=self._deconvolution_mode, predefined_covariance=self._predefined_covariance)
        
        ## Reconstruct volumes for all alphas
        regularization_parameter_estimator.run_reconstructions()

        ## Append filename pointing to computed results stored in txt-file
        self._filenames.append(regularization_parameter_estimator.get_filename_of_txt_file())


    ##
    #       Reconstruct the volumes based on first-order Tikhonov
    #             regularization by sweeping through all alphas specified in
    #             alpha_array
    # \date       2016-08-02 00:20:20+0100
    #
    # \param      self  The object
    #
    def _run_reconstructions_TK1(self):

        ## Initialize first-order Tikhonov solver
        regularization_parameter_estimator = tkrpe.TikhonovRegularizationParameterEstimator(self._stacks, self._HR_volume_init, alpha_cut=self._alpha_cut, iter_max=self._iter_max, alpha_array=self._alpha_array, dir_results=self._dir_results, reg_type="TK1", minimizer=self._minimizer, deconvolution_mode=self._deconvolution_mode, predefined_covariance=self._predefined_covariance)
        
        ## Reconstruct volumes for all alphas
        regularization_parameter_estimator.run_reconstructions()

        ## Append filename pointing to computed results stored in txt-file
        self._filenames.append(regularization_parameter_estimator.get_filename_of_txt_file())


    ##
    #       Reconstruct the volumes based on TV-L2 regularization by
    #             sweeping through all alphas specified in alpha_array
    # \date       2016-08-02 00:20:20+0100
    #
    # \param      self  The object
    #
    def _run_reconstructions_TVL2(self):

        ## Estimate inital value based on TK1-regularization step prior ADMM algorithm
        ## \post self._HR_volume_init is updated
        SRR = tk.TikhonovSolver(self._stacks, self._HR_volume_init, alpha_cut=self._alpha_cut, alpha=0.05, iter_max=10, reg_type="TK1", minimizer=self._minimizer, deconvolution_mode=self._deconvolution_mode, predefined_covariance=self._predefined_covariance)
        SRR.run_reconstruction()

        ## Initialize TV-L2 solver
        regularization_parameter_estimator = tvl2rpe.TVL2RegularizationParameterEstimator(self._stacks, self._HR_volume_init, alpha_cut=self._alpha_cut, iter_max=self._iter_max, alpha_array=self._alpha_array, dir_results=self._dir_results, ADMM_iterations=self._ADMM_iterations, rho_array=self._rho_array, ADMM_iterations_output_dir=self._ADMM_iterations_output_dir, ADMM_iterations_output_filename_prefix=self._ADMM_iterations_output_filename_prefix, minimizer=self._minimizer, deconvolution_mode=self._deconvolution_mode, predefined_covariance=self._predefined_covariance)

        ## Reconstruct volumes for all alphas and rhos
        regularization_parameter_estimator.run_reconstructions(estimate_initial_value=False)

        ## Append filename pointing to computed results stored in txt-file
        self._filenames.append(regularization_parameter_estimator.get_filename_of_txt_file())



    ##
    #       Print information on screen for zeroth-order Tikhonov
    # \date       2016-08-02 00:54:55+0100
    #
    # \param      self  The object
    # \param      data  Data array containing the information on performed
    #                   recontruction
    #
    def _print_data_on_screen_TK0(self, data):
        TK_approach = "TK0"
        self._print_data_on_screen_TK(data, TK_approach)


    ##
    #       Print information on screen for first-order Tikhonov
    # \date       2016-08-02 00:54:55+0100
    #
    # \param      self  The object
    # \param      data  Data array containing the information on performed
    #                   recontruction
    #
    def _print_data_on_screen_TK1(self, data):
        TK_approach = "TK1"
        self._print_data_on_screen_TK(data, TK_approach)


    ##
    #       Print information on screen for zeroth/first-order Tikhonov
    # \date       2016-08-02 00:54:55+0100
    #
    # \param      self         The object
    # \param      data         Data array containing the information on
    #                          performed recontruction
    # \param      TK_approach  Either "TK0" or "TK1"
    #
    def _print_data_on_screen_TK(self, data, TK_approach):

        alpha_ids = data[:,0]
        alphas = data[:,1]
        residuals_data_fit = data[:,2]
        residuals_prior = data[:,3]

        print("\n\t--- %s-Regularization ---" %(TK_approach))
        print("\t\t#\talpha\t\tResidual data fit\tPrior Psi")
        for i in range(0, len(alphas)):
            print("\t\t%d\t%s\t\t%.3e\t\t%.3e" %(alpha_ids[i], alphas[i], residuals_data_fit[i], residuals_prior[i]))


    ##
    #       Print information on screen for TV-L2
    # \date       2016-08-02 00:54:55+0100
    #
    # \param      self  The object
    # \param      data  Data array containing the information on performed
    #                   recontruction
    #
    def _print_data_on_screen_TVL2(self, data):

        rho_ids = data[:,0]
        alpha_ids = data[:,1]
        rhos = data[:,2]
        alphas = data[:,3]
        residuals_data_fit = data[:,4]
        residuals_prior = data[:,5]

        print("\n\t--- TV-L2-Regularization ---")
        print("\t#\trho\talpha\t\tResidual data fit\tPrior Psi")
        for i in range(0, len(alphas)):
            print("\t(%d,%d)\t%s\t%s\t\t%.3e\t\t%.3e" %(rho_ids[i], alpha_ids[i], rhos[i], alphas[i], residuals_data_fit[i], residuals_prior[i]))


    ##
    #       Add curve on graph based on data from zeroth-order Tikhonov
    #             reconstruction
    # \date       2016-08-02 01:02:02+0100
    #
    # \param      self    The object
    # \param      data    Data array containing information on performed reconstruction
    # \param      i_file  Counter to have id for curve
    # \param      ax      Axis handle of figure
    #
    def _plot_add_curve_TK0(self, data, i_file, ax):
        TK_approach = "TK0"
        self._plot_add_curve_TK(data, i_file, ax, TK_approach)


    ##
    #       Add curve on graph based on data from first-order Tikhonov
    #             reconstruction
    # \date       2016-08-02 01:02:02+0100
    #
    # \param      self    The object
    # \param      data    Data array containing information on performed reconstruction
    # \param      i_file  Counter to have id for curve
    # \param      ax      Axis handle of figure
    #
    def _plot_add_curve_TK1(self, data, i_file, ax):
        TK_approach = "TK1"
        self._plot_add_curve_TK(data, i_file, ax, TK_approach)


    ##
    #       Add curve on graph based on data from zeroth/first-order
    #             Tikhonov reconstruction
    # \date       2016-08-02 01:02:02+0100
    #
    # \param      self         The object
    # \param      data         Data array containing information on performed
    #                          reconstruction
    # \param      i_file       Counter to have id for curve
    # \param      ax           Axis handle of figure
    # \param      TK_approach  string, either "TK0" or "TK1"
    #
    def _plot_add_curve_TK(self, data, i_file, ax, TK_approach):

        alpha_ids = data[:,0]
        alphas = data[:,1]
        residuals_data_fit = data[:,2]
        residuals_prior = data[:,3]

        ax.plot(residuals_data_fit, residuals_prior, self._PLOT_FORMAT[i_file], label=TK_approach)


    ##
    #       Add curve on graph based on data from TV-L2 reconstruction
    # \date       2016-08-02 01:02:02+0100
    #
    # \param      self         The object
    # \param      data         Data array containing information on performed
    #                          reconstruction
    # \param      i_file       Counter to have id for curve
    # \param      ax           Axis handle of figure
    #
    def _plot_add_curve_TVL2(self, data, i_file, ax):

        rho_ids = data[:,0]
        alpha_ids = data[:,1]

        rhos = data[:,2]
        rhos_unique = np.unique(rhos)

        alphas = data[:,3]
        alphas_unique = np.unique(alphas)

        residuals_data_fit = data[:,4]
        residuals_prior = data[:,5]

        for i in range(0,len(rhos_unique)):
            rho = rhos_unique[i]
            ind = np.where(rhos==rho)
            label = "TV-L2 (rho=" + str(rho) + ")"
            ax.plot(residuals_data_fit[ind], residuals_prior[ind], self._PLOT_FORMAT[i_file], label=label)

            i_file += 1


    ##
    #       Add information on figure
    # \date       2016-08-02 01:06:15+0100
    #
    # \param      self  The object
    # \param      ax    Axis handle of figure
    #
    def _plot_add_figure_information(self, ax):
        ## Show grid
        ax.grid()

        ## Add legend
        legend = ax.legend(loc="upper right")

        plt.title("L-curve\n$\Phi(\\vec{x}) = \\frac{1}{2}\sum_{k=1}^K \Vert M_k (\\vec{y}_k - A_k \\vec{x}) \Vert_{\ell^2}^2 + \\alpha \Psi(\\vec{x})\\rightarrow \min$" )
        # plt.title("L-curve for " + regularization_types[i_reg_type] + "Regularization")
        # plt.ylabel("Prior term $\\displaystyle\Psi(\\vec{x}_\\alpha)$")
        plt.ylabel("Prior term $\Psi(\\vec{x}_\\alpha)$")
        plt.xlabel("Residual data fit $\sum_{k=1}^K \Vert M_k (\\vec{y}_k - A_k \\vec{x}_\\alpha) \Vert_{\ell^2}^2$")

        plt.draw()
        plt.pause(0.5) ## important! otherwise fig is not shown.

