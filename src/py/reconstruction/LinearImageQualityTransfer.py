##-----------------------------------------------------------------------------
# \file LinearImageQualityTransfer.py
# \brief      Class deploying a linear model for image quality transfer
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear
from scipy.optimize import nnls
import time
from datetime import timedelta

## Import modules
import utilities.SimpleITKHelper as sitkh


class LinearImageQualityTransfer(object):


    def __init__(self, stack, references):
        