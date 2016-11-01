#------------------------------------------------------------------------------
# \file RegularizationParameterEstimator.py
# \brief      This is the basis class containing all common attributs/functions
#             for TikhonovRegularizationParameterEstimator and
#             TVL2RegularizationParameterEstimator.
#
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

## Import modules from src-folder
import utilities.SimpleITKHelper as sitkh
import base.PSF as psf
import base.Stack as st
import reconstruction.InverseProblemSolver as ips


class RegularizationParameterEstimator(object):

    ##-------------------------------------------------------------------------
    # \brief         Constructor
    # \date          2016-08-01 23:40:34+0100
    #
    # \param         self                     The object
    # \param[in]     stacks                   list of Stack objects containing
    #                                         all stacks used for the
    #                                         reconstruction
    # \param[in,out] HR_volume                Stack object containing the
    #                                         current estimate of the HR volume
    #                                         (used as initial value + space
    #                                         definition)
    # \param[in]     alpha_cut                Cut-off distance for Gaussian
    #                                         blurring filter
    # \param[in]     iter_max                 number of maximum iterations,
    #                                         scalar
    # \param[in]     alpha_array              array containing regularization
    #                                         parameter to sweep through, list
    # \param[in]     dir_results              Directory to store computed
    #                                         results. If 'None' no results are
    #                                         written.
    # \param[in]     filename_results_prefix  Prefix applied for each filename
    #                                         written to dir_results
    #
    def __init__(self, stacks, HR_volume, alpha_cut=3, iter_max=10, alpha_array=[None], dir_results="RegularizationParameterEstimation/", filename_results_prefix=""):

        ## Initialize variables
        self._stacks = stacks
        self._HR_volume = HR_volume
        self._N_stacks = len(stacks)

        self._alpha_cut = alpha_cut
        self._iter_max = iter_max
        self._alpha_array = alpha_array

        ## Parameters for output
        self._dir_results = dir_results
        self._filename_results_prefix = filename_results_prefix


    ##-------------------------------------------------------------------------
    # \brief      Sets the directory results.
    # \date       2016-08-01 16:30:57+0100
    #
    # \param      self         The object
    # \param[in]  dir_results  string
    #
    def set_directory_results(self, dir_results):
        self._dir_results = dir_results


    ##-------------------------------------------------------------------------
    # \brief      Gets the directory results.
    # \date       2016-08-01 16:32:14+0100
    #
    # \param      self  The object
    #
    # \return     The directory results.
    #
    def get_directory_results(self):
        return self._dir_results


    ##-------------------------------------------------------------------------
    # \brief      Set prefix for all output results written to
    #             directory_results.
    # \date       2016-08-01 16:29:48+0100
    #
    # \param      self             The object
    # \param[in]  filename_prefix  string
    #
    def set_filename_results_directory(self, filename_prefix):
        self._filename_results_prefix = filename_prefix


    ##-------------------------------------------------------------------------
    # \brief      Gets the filename results directory.
    # \date       2016-08-01 16:29:33+0100
    #
    # \param      self  The object
    #
    # \return     The filename results directory.
    #
    def get_filename_results_directory(self):
        return self._filename_results_prefix


    ##-------------------------------------------------------------------------
    # \brief      Get the filename of text file.
    # \date       2016-08-01 18:31:26+0100
    #
    # \param      self  The object
    #
    # \return     The filename of text file as list.
    #
    def get_filename_of_txt_file(self):
        return self._filename_of_txt_file


    ##-------------------------------------------------------------------------
    # \brief      Sets the alpha array.
    #
    # Define array of regularization parameters which will be used for the
    # computation
    # \date       2016-08-01 16:33:30+0100
    #
    # \param      self         The object
    # \param[in]  alpha_array  numpy array of alpha values
    #
    def set_alpha_array(self, alpha_array):
        self._alpha_array = alpha_array


    ##-------------------------------------------------------------------------
    # \brief      Gets the alpha array.
    # \date       2016-08-01 16:34:06+0100
    #
    # \param      self  The object
    #
    # \return     numpy array of alpha values
    #
    def get_alpha_array(self):
        return self._alpha_array


    ##-------------------------------------------------------------------------
    # \brief      Sets the maximum number of iterations for Tikhonov solver.
    # \date       2016-08-01 16:35:09+0100
    #
    # \param      self      The object
    # \param[in]  iter_max  number of maximum iterations, scalar
    #
    def set_iter_max(self, iter_max):
        self._iter_max = iter_max


    
    def _create_file(self, filename, header):
        file_handle = open(self._dir_results + filename + ".txt", "w")
        file_handle.write(header)
        file_handle.close()


    def _write_array_to_file(self, filename, array, format="%.10e", delimiter="\t"):
        file_handle = open(self._dir_results + filename + ".txt", "a")
        np.savetxt(file_handle, array, fmt=format, delimiter=delimiter)
        file_handle.close()



    def _fitting_curve(self, x, a,b,c):

        # foo = np.polyval([a,b,c,d],x)
        foo = a*np.exp(-b*x) + c

        return foo

    def _get_maximum_curvature_point(self, x, y):

        # scale = x.mean()
        scale = 1

        x = x/scale
        y = y/scale

        N_points = len(x)
        radius2 = np.zeros(N_points-2)

        M = np.zeros((3,3))

        for i in range(1, N_points-1):
            M[:,0] = 1
            M[:,1] = x[i-1:i+2]
            M[:,2] = y[i-1:i+2]
            b = x[i-1:i+2]**2 + y[i-1:i+2]**2

            [A, B, C] = np.linalg.solve(M,b)

            radius2[i-1] = A + (B**2 + C**2)/4
            print("(xm, ym, r2) = (%s, %s, %s)" %(B/2.,C/2.,radius2[i-1]))

        i_max_curvature = np.argmin(radius2)+1

        return i_max_curvature
