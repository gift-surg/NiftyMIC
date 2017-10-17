##
# \file ParameterNormalization.py
# \brief      Class containing functions to normalize parameters. This can be
#             used to normalize parameters used for optimization e.g.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
#


## Import libraries
import sys
# import SimpleITK as sitk
# import itk
import numpy as np

## Import modules
import pythonhelper.SimpleITKHelper as sitkh


##
#       Class to normalize parameters
# \date       2016-11-10 17:05:30+0000
#
class ParameterNormalization(object):

    ##
    #       Constructor
    # \date       2016-11-10 17:08:25+0000
    #
    # \param      self              The object
    # \param      parameters_array  (N x N_param)-np.array. N_param
    #                               different types of parameters and N amount
    #                               of each of them.
    #
    def __init__(self, parameters_array):

        ## Create copy of parameters
        self._parameters_array = np.array(parameters_array)

        ## Amount of different parameters
        self._N_parameters = self._parameters_array.shape[1]

        self._coefficients = 1.* np.concatenate((np.zeros((1,self._N_parameters)), np.ones((1,self._N_parameters))))


    ##
    #       Gets the normalization coefficients as (2 x
    #             N_param)-np.array.
    # \date       2016-11-10 18:00:52+0000
    #
    # The first row denotes the mean and the second the computed standard
    # deviation of the originally provided parameter array.
    #
    # \param      self  The object
    #
    # \return     The normalization coefficients as (2 x N_param)-np.array.
    #
    def get_normalization_coefficients(self):
        return np.array(self._coefficients)


    ##
    #       Calculates the normalization coefficients which will be used
    #             for normalization and denormalization routines.
    # \date       2016-11-10 18:01:17+0000
    #
    # \param      self  The object
    #
    def compute_normalization_coefficients(self):
        
        coefficients = np.zeros((2, self._N_parameters))

        for i in range(0, self._N_parameters):
            coefficients[0,i] = np.mean(self._parameters_array[:,i])

            sigma = np.std(self._parameters_array[:,i])
            if abs(sigma) < 1e-8:
                coefficients[1,i] = 1.
            else:
                coefficients[1,i] = sigma

        self._coefficients = coefficients


    ##
    #       Normalize parameters based on previously computed
    #             coefficients.
    # \date       2016-11-10 18:04:05+0000
    #
    # \remark I would like to not make a copy of the parameters
    #
    # \param      self        The object
    # \param      parameters  (N x N_params)-np.array to be normalized
    #
    # \return     normalized parameter array
    #
    def normalize_parameters(self, parameters):

        parameters = np.array(parameters)

        ## Compute p_norm = (p - mean)/std
        for i in range(0, self._N_parameters):
            parameters[:,i] = (parameters[:,i] - self._coefficients[0,i])/self._coefficients[1,i]

        return parameters


    ##
    #       Denormalize parameters based on previously computed
    #             coefficients.
    # \date       2016-11-10 18:05:49+0000
    #
    # \param      self        The object
    # \param      parameters  (N x N_params)-np.array to be normalized
    #
    # \return     denormalized parameter array
    #
    def denormalize_parameters(self, parameters):

        parameters = np.array(parameters)

        ## Compute p = p_norm*std + mean
        for i in range(0, self._N_parameters):
            parameters[:,i] = parameters[:,i]*self._coefficients[1,i] + self._coefficients[0,i]

        return parameters

