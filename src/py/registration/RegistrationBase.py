#!/usr/bin/python

##-----------------------------------------------------------------------------
# \file RegistrationBase.py
# \brief      Basis attributes and member functions all registration methods
#             should share
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016


## Import libraries
from abc import ABCMeta, abstractmethod
import sys
import SimpleITK as sitk
import itk
import numpy as np
import time
from datetime import timedelta
from scipy.optimize import least_squares

## Import modules
import utilities.SimpleITKHelper as sitkh
import base.Stack as st


##-----------------------------------------------------------------------------
# \brief      Basis class containing the shared attributes and functions
# \date       2016-11-06 16:58:15+0000
#
class RegistrationBase(object):
    __metaclass__ = ABCMeta

    ##-------------------------------------------------------------------------
    # \brief      Constructor
    # \date       2016-11-06 16:58:43+0000
    #
    # \param      self             The object
    # \param      fixed            The fixed
    # \param      moving           The moving
    # \param      use_fixed_mask   The use fixed mask
    # \param      use_moving_mask  The use moving mask
    # \param      use_verbose      The use verbose
    #
    def __init__(self, fixed=None, moving=None, use_fixed_mask=False, use_moving_mask=False, use_verbose=False):

        ## Set Fixed and moving stacks
        if fixed is not None:
            self._fixed = st.Stack.from_stack(fixed)
        else:
            self._fixed = None

        if moving is not None:
            self._moving = st.Stack.from_stack(moving)
        else:
            self._moving = None
    
        ## Set booleans to use mask            
        self._use_fixed_mask = use_fixed_mask
        self._use_moving_mask = use_moving_mask

        ## Verbose computation
        self._use_verbose = use_verbose


    ##-------------------------------------------------------------------------
    # \brief      Sets fixed/reference/target image.
    # \date       2016-11-06 16:59:14+0000
    #
    # \param      self   The object
    # \param      fixed  fixed/reference/target image as Stack object
    #
    def set_fixed(self, fixed):
        self._fixed = st.Stack.from_stack(fixed)

    ##-------------------------------------------------------------------------
    # \brief      Gets fixed/reference/target image.
    # \date       2016-11-06 17:00:18+0000
    #
    # \return     The fixed/reference/target image as Stack object.
    #
    def get_fixed(self):
        return self._fixed


    ##-------------------------------------------------------------------------
    # \brief      Sets moving/floating/source image.
    # \date       2016-11-06 17:00:50+0000
    #
    # \param      self    The object
    # \param      moving  moving/floating/source image as Stack object
    #
    def set_moving(self, moving):
        self._moving = st.Stack.from_Stack(moving)

    ##-------------------------------------------------------------------------
    # \brief      Gets moving/floating/source image.
    # \date       2016-11-06 17:02:16+0000
    #
    # \param      self  The object
    #
    # \return     The moving/floating/source image as Stack object.
    #
    def get_moving(self):
        return self._moving


    ##-------------------------------------------------------------------------
    # \brief      Specify whether mask of fixed image shall be used for
    #             registration
    # \date       2016-11-06 17:03:05+0000
    #
    # \param      self  The object
    # \param      flag  The flag as boolean
    #
    def use_fixed_mask(self, flag):
        self._use_fixed_mask = flag


    ##-------------------------------------------------------------------------
    # \brief      Specify whether mask of moving image shall be used for
    #             registration
    # \date       2016-11-06 17:03:05+0000
    #
    # \param      self  The object
    # \param      flag  The flag as boolean
    #
    def use_moving_mask(self, flag):
        self._use_moving_mask = flag


    ##-------------------------------------------------------------------------
    # \brief      Specify whether output information shall be produced.
    # \date       2016-11-06 17:07:01+0000
    #
    # \param      self  The object
    # \param      flag  The flag
    #
    def use_verbose(self, flag):
        self._use_verbose = flag


    ##-------------------------------------------------------------------------
    # \brief      Gets the parameters estimated by registration algorithm.
    # \date       2016-11-06 17:05:38+0000
    #
    # \param      self  The object
    #
    # \return     The parameters.
    #
    @abstractmethod
    def get_parameters(self):
        pass


    ##-------------------------------------------------------------------------
    # \brief      Gets the registraton transform sitk.
    # \date       2016-11-06 17:10:14+0000
    #
    # \param      self  The object
    #
    # \return     The registraton transform sitk.
    #
    @abstractmethod
    def get_registration_transform_sitk(self):
        pass


    ##-------------------------------------------------------------------------
    # \brief      Print statistics associated to performed registration
    # \date       2016-11-06 17:07:56+0000
    #
    # \param      self  The object
    #
    def print_statistics(self):
        # print("\nStatistics for performed registration:" %(self._reg_type))
        # if self._elapsed_time_sec < 0:
        #     raise ValueError("Error: Elapsed time has not been measured. Run 'run_reconstruction' first.")
        # else:
        print("\tElapsed time = %s" %(self._elapsed_time))
        # print("\tell^2-residual sum_k ||M_k(A_k x - y_k||_2^2 = %.3e" %(self._residual_ell2))
        # print("\tprior residual = %.3e" %(self._residual_prior))


