##
# \file MotionSimulator.py
# \brief      Abstract class to define interface for motion simulator
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2017
#

# Import libraries
import os
import numpy as np
import SimpleITK as sitk
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph


class MotionSimulator(object):
    __metaclass__ = ABCMeta

    def __init__(self, dimension, verbose):
        self._transforms_sitk = None
        self._dimension = dimension
        self._verbose = verbose

    @abstractmethod
    def simulate_motion(self):
        pass

    def get_transforms_sitk(self):

        # Return copy of transforms
        transforms_sitk = [
            eval("sitk." + self._transforms_sitk[0].GetName() +
                 "(t)") for t in self._transforms_sitk
        ]
        return transforms_sitk

    def write_transforms_sitk(self,
                              directory,
                              prefix_filename="Euler3dTransform_"):
        ph.create_directory(directory)
        for i, transform in enumerate(self._transforms_sitk):
            path_to_file = os.path.join(
                directory, "%s%d.tfm" % (prefix_filename, i))
            sitk.WriteTransform(transform, path_to_file)
            if self._verbose:
                ph.print_info("Transform written to %s" % path_to_file)


class RigidMotionSimulator(MotionSimulator):
    __metaclass__ = ABCMeta

    def __init__(self, dimension, verbose):
        MotionSimulator.__init__(self, dimension=dimension, verbose=verbose)
        self._transform0_sitk = eval("sitk.Euler%dDTransform" % (dimension))


class RandomRigidMotionSimulator(RigidMotionSimulator):

    def __init__(self,
                 dimension,
                 angle_max_deg=5,
                 translation_max=5,
                 verbose=False):
        RigidMotionSimulator.__init__(
            self, dimension=dimension, verbose=verbose)
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
