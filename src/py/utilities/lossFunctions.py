##
#  \file lossFunctions.py
#  \brief Definition of loss functions similar according to
#  \p https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.least_squares.html
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2017

import numpy as np

def linear(e):
    return e

def gradient_linear(e):
    return np.ones_like(e).astype(np.float64)

def soft_l1(e):
    return 2. * (np.sqrt(1.+e) - 1.)

def gradient_soft_l1(e):
    return 1. / np.sqrt(1.+e)

def huber(e):
    return np.where(e>1., 2.*np.sqrt(e)-1., e)

def gradient_huber(e):
    return np.where(e>1., 1./np.sqrt(e), 1.)