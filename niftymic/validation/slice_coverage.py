##
# \file slice_coverage.py
# \brief      Class to visualize slice coverage over reconstruction space
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Feb 2019
#

import numpy as np
import SimpleITK as sitk


##
# Class to visualize slice coverage over reconstruction space
# \date       2019-02-23 21:17:49+0000
#
class SliceCoverage(object):

    def __init__(self, stacks, reconstruction_sitk):
        self._stacks = stacks
        self._reconstruction_sitk = reconstruction_sitk

        self._coverage_sitk = None

    ##
    # Gets the slice coverage as Image. The (integer) intensity values reflect
    # the number of slices that have contributed to this particular voxel.
    # \date       2019-02-23 21:18:10+0000
    #
    # \param      self  The object
    #
    # \return     Slice coverage as sitk.Image uint8 image.
    #
    def get_coverage_sitk(self):
        if self._coverage_sitk is None:
            raise RuntimeError("Execute 'run' first")
        return sitk.Image(self._coverage_sitk)

    ##
    # Compute slice coverage
    # \date       2019-02-23 21:19:55+0000
    #
    # \param      self  The object
    #
    def run(self):

        # create zero image
        coverage_sitk = sitk.Image(self._reconstruction_sitk) * 0

        for i, stack in enumerate(self._stacks):
            print("Slices of stack %d/%d ... " % (i + 1, len(self._stacks)))

            # Add each individual slice contribution
            for slice in stack.get_slices():
                coverage_sitk = self._add_slice_contribution(
                    slice, coverage_sitk)

        # Cast to unsigned integer
        self._coverage_sitk = sitk.Cast(coverage_sitk, sitk.sitkUInt8)

    ##
    # Adds a slice contribution.
    # \date       2019-02-23 21:27:12+0000
    #
    # \param      slice          Slice as sl.Slice object
    # \param      coverage_sitk  sitk.Image reflecting the current iteration of
    #                            slice coverage
    #
    # \return     Updated slice contribution, sitk.Image
    #
    @staticmethod
    def _add_slice_contribution(slice, coverage_sitk):

        #
        slice_sitk = sitk.Image(slice.sitk)
        spacing = np.array(slice_sitk.GetSpacing())
        spacing[-1] = slice.get_slice_thickness()
        slice_sitk.SetSpacing(spacing)

        contrib_nda = sitk.GetArrayFromImage(slice_sitk)
        contrib_nda[:] = 1
        contrib_sitk = sitk.GetImageFromArray(contrib_nda)
        contrib_sitk.CopyInformation(slice_sitk)

        coverage_sitk += sitk.Resample(
            contrib_sitk,
            coverage_sitk,
            sitk.Euler3DTransform(),
            sitk.sitkNearestNeighbor,
            0,
            coverage_sitk.GetPixelIDValue(),
        )

        return coverage_sitk
