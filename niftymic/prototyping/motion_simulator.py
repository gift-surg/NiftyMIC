##
# \file MotionSimulator.py
# \brief      Abstract class to define interface for motion simulator
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2017
#

# Import libraries
from abc import ABCMeta, abstractmethod
import SimpleITK as sitk
import numpy as np


class MotionSimulator(object):
    __metaclass__ = ABCMeta

    def __init__(self, dimension):
        self._transforms_sitk = None
        self._dimension = dimension

    @abstractmethod
    def simulate_motion(self):
        pass

    def get_transforms_sitk(self):
        # Return copy of transforms

        transforms_sitk = [eval("sitk." + self._transforms_sitk[0].GetName() +
                                "(t)") for t in self._transforms_sitk]
        return transforms_sitk


class RigidMotionSimulator(MotionSimulator):
    __metaclass__ = ABCMeta

    def __init__(self, dimension):
        MotionSimulator.__init__(self, dimension)
        self._transform0_sitk = eval("sitk.Euler%dDTransform" % (dimension))


class RandomRigidMotionSimulator(RigidMotionSimulator):

    def __init__(self, dimension, angle_max_deg=5, translation_max=5):
        RigidMotionSimulator.__init__(self, dimension)
        self._angle_max_deg = angle_max_deg
        self._translation_max = translation_max

    def simulate_motion(self, seed=None, simulations=1):

        np.random.seed(seed)

        # Create random translation \f$\in\f$ [\p -translation_max, \p
        # translation_max]
        translation = 2. * np.random.rand(simulations, self._dimension) * \
            self._translation_max - self._translation_max

        # Create random rotation \f$\in\f$ [\p -angle_deg_max, \p
        # angle_deg_max]
        angle_rad = \
            (2. * np.random.rand(simulations, self._dimension) * self._angle_max_deg -
                self._angle_max_deg) * np.pi / 180.

        # Set resulting rigid motion transform
        self._transforms_sitk = [None] * simulations

        for i in range(simulations):
            self._transforms_sitk[i] = self._transform0_sitk()

            parameters = list(angle_rad[i, :])
            parameters.extend(translation[i, :])

            self._transforms_sitk[i].SetParameters(parameters)
