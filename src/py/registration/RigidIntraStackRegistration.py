#!/usr/bin/python

##-----------------------------------------------------------------------------
# \file RigidIntraStackRegistration.py
# \brief      Abstract class used for intra-stack registration steps. Slices
#             are only transformed in-plane. Hence, only 2D transforms are
#             applied
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
#


## Import libraries
import sys
import SimpleITK as sitk
import itk
import numpy as np
from scipy.optimize import least_squares
import time
from datetime import timedelta

## Import modules
import base.PSF as psf
import base.Slice as sl
import base.Stack as st
import utilities.SimpleITKHelper as sitkh
from registration.IntraStackRegistration import IntraStackRegistration


class RigidIntraStackRegistration(IntraStackRegistration):

    def __init__(self, stack=None, reference=None, use_stack_mask=False, use_reference_mask=False, use_verbose=False, initializer_type="identity", interpolator="Linear", alpha_neighbour=1, alpha_reference=1, alpha_parameter=1):

        ## Parameters specific to rigid transform
        self._transform_type_sitk_new = sitk.Euler2DTransform()
        self._transform_type_dofs = 3
        self._optimization_dofs = 3