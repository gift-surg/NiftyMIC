##
# \file patch_generator.py
# \brief      Class to generate patches from slices of a stack
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

import itertools
import numpy as np
import SimpleITK as sitk
from abc import ABCMeta, abstractmethod
import pysitk.python_helper as ph

import niftymic.base.stack as st


class PatchGenerator(object):
    __metaclass__ = ABCMeta

    def __init__(self, stack):
        self._stack = st.Stack.from_stack(stack)

        self._computational_time = ph.get_zero_time()
        self._patches = None

    def set_stack(self, stack):
        self._stack = st.Stack.from_stack(stack)

    def get_stack(self):
        return st.Stack.from_stack(self._stack)

    def generate_patches(self):
        time_start = ph.start_timing()

        self._patches = []
        self._generate_patches()

        # Get computational time
        self._computational_time = ph.stop_timing(time_start)

    ##
    # Gets the computational time it took to perform the registration
    # \date       2017-08-08 16:59:45+0100
    #
    # \param      self  The object
    #
    # \return     The computational time.
    #
    def get_computational_time(self):
        return self._computational_time

    def get_patches(self):
        return [st.Stack.from_stack(patch) for patch in self._patches]

    @abstractmethod
    def _generate_patches(self):
        pass


class SquaredPatchGenerator(PatchGenerator):

    def __init__(self, stack, patch_size, tolerance=0.4, unit="voxel"):

        PatchGenerator.__init__(self, stack=stack)
        self._patch_size = patch_size
        self._tolerance = tolerance
        self._unit = unit

    def _generate_patches(self):

        if self._unit != "voxel":
            spacing = float(self._stack.sitk.GetSpacing()[0])
            patch_size = int(self._patch_size / spacing)
        else:
            patch_size = self._patch_size

        stack = self._stack.get_cropped_stack_based_on_mask()

        nda = sitk.GetArrayFromImage(stack.sitk)
        shape = nda.shape

        intervals_x = range(0, shape[2], patch_size)
        intervals_y = range(0, shape[1], patch_size)

        intervals_x.append(shape[2])
        intervals_y.append(shape[1])

        for i, x in enumerate(intervals_x[0:-1]):
            xp1 = intervals_x[i+1]
            for j, y in enumerate(intervals_y[0:-1]):
                yp1 = intervals_y[j+1]
                patch_sitk = stack.sitk[x:xp1, y:yp1, :]
                patch_sitk_mask = stack.sitk_mask[x:xp1, y:yp1, :]

                name = stack.get_filename() + "_%d_%d" % (i, j)
                patch = st.Stack.from_sitk_image(
                    image_sitk=patch_sitk,
                    filename=name,
                    image_sitk_mask=patch_sitk_mask)

                patch = self._get_valid_patch(patch)
                if patch is None:
                    continue

                self._patches.append(patch)

    def _get_valid_patch(self, patch):
        patch = patch.get_cropped_stack_based_on_mask()

        if patch is None:
            return None

        eliminate_slices = []
        for i in range(patch.get_number_of_slices()):
            slice = patch.get_slice(i)
            nda_mask = sitk.GetArrayFromImage(slice.sitk_mask)
            proportion = nda_mask.sum() / float(nda_mask.size)

            if proportion < self._tolerance:
                eliminate_slices.append(i)

        for i in eliminate_slices[::-1]:
            patch.eliminate_slice(i)

        return patch