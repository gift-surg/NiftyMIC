# \file MotionSimulator.py
#  \brief Abstract class to define interface for motion simulator
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2017


# Import libraries
from abc import ABCMeta, abstractmethod
import SimpleITK as sitk
import numpy as np


class MotionSimulator(object):
    __metaclass__ = ABCMeta

    def __init__(self, dimension):
        self._transform_sitk = None
        self._dimension = dimension

    @abstractmethod
    def simulate_motion(self):
        pass

    def get_transform_sitk(self):
        # Return copy of transform
        transform_sitk = eval("sitk." + self._transform_sitk.GetName() +
                              "(self._transform_sitk)")
        return transform_sitk


class RigidMotionSimulator(MotionSimulator):
    __metaclass__ = ABCMeta

    def __init__(self, dimension):
        MotionSimulator.__init__(self, dimension)
        self._transform_sitk = eval("sitk.Euler%dDTransform()" % (dimension))


class RandomRigidMotionSimulator(RigidMotionSimulator):

    def __init__(self, dimension, angle_max_deg=5, translation_max=5):
        RigidMotionSimulator.__init__(self, dimension)
        self._angle_max_deg = angle_max_deg
        self._translation_max = translation_max

    def simulate_motion(self, seed=None):

        np.random.seed(seed)

        # Create random translation \f$\in\f$ [\p -translation_max, \p
        # translation_max]
        translation = 2. * np.random.rand(self._dimension) * \
            self._translation_max - self._translation_max

        # Create random rotation \f$\in\f$ [\p -angle_deg_max, \p
        # angle_deg_max]
        angle_rad = \
            (2. * np.random.rand(self._dimension) * self._angle_max_deg -
                self._angle_max_deg) * np.pi / 180.

        # Set resulting rigid motion transform
        parameters = list(angle_rad)
        parameters.extend(translation)
        self._transform_sitk.SetParameters(parameters)
