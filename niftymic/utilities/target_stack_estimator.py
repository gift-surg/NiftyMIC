##
# \file target_stack_estimator.py
# \brief      Class to estimate target stack automatically
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       January 2018
#


import os
import re
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st


##
# Class to estimate target stack automatically
# \date       2018-01-26 16:32:11+0000
#
class TargetStackEstimator(object):

    def __init__(self):
        self._target_stack_index = None

        self._compute_motion_score = {

            # implementation according to github fetalReconstruction
            "github": self._compute_motion_score_github,

            # implementation according to TMI paper Kainz2015
            "kainz2015": self._compute_motion_score_kainz2015,
        }
        # appears more meaningful to me (see comments below)
        self._mode = "kainz2015"

        self._computational_time = ph.get_zero_time()

    def get_target_stack_index(self):
        return self._target_stack_index

    def get_computational_time(self):
        return self._computational_time

    @staticmethod
    def _compute_volume(file_path):
        mask_sitk = sitkh.read_nifti_image_sitk(str(file_path), sitk.sitkUInt8)

        # Compute mask volume
        mask_nda = sitk.GetArrayFromImage(mask_sitk)
        spacing = np.array(mask_sitk.GetSpacing())
        volume = np.sum(mask_nda) * spacing.prod()

        return volume

    @staticmethod
    def _compute_singular_values(stack):

        # (z,y,x) array
        nda = sitk.GetArrayFromImage(stack.sitk)

        # reshape to M x K, M number of slice pixels, K number of slices
        A = nda.reshape(nda.shape[0], -1).transpose()

        U, s, Vt = np.linalg.svd(A)

        return s

    ##
    # Calculates the motion score similar to TMI Kainz2015 paper. Instead, a
    # relative rank is used in order to not penalize stacks with more slices.
    # (Motion score: A lower score indicates less motion).
    # @date       2019-08-31 16:01:12+0100
    #
    # @param      sing_values  singular values, vector
    # @param      threshold    error threshold, float
    #
    # @return     The motion score, float.
    #
    @staticmethod
    def _compute_motion_score_kainz2015(
            sing_values, threshold=np.sqrt(1 - 0.99**2)):

        sing_values_2 = np.square(sing_values)

        # compute Frobenius norm
        A_norm = np.sum(sing_values_2)

        # compute relative rank-approximation error
        # the higher r, the smaller the delta_r
        delta_r = np.sqrt(np.array([
            np.sum(sing_values_2[r + 1:]) / A_norm
            for r in range(len(sing_values_2))
        ]))

        # find lowest rank approximation that leads to error < threshold
        r = np.where(delta_r < threshold)[0][0] + 1

        # use relative rank so as to not "penalise" stack with more slices
        r_rel = r / float(len(sing_values))

        # surrogate motion score, the lower the better
        motion_score = r_rel * delta_r[r - 1]

        return motion_score

    ##
    # Calculates the motion score based on TMI Kainz2015 paper, but the version
    # as found on the GitHub fetalReconstruction repo. (A lower motion score
    # shall indicate less motion).
    # @date       2019-08-31 16:01:12+0100
    #
    # @param      sing_values  singular values, vector
    # @param      threshold    error threshold, float
    #
    # @return     The motion score, float.
    #
    @staticmethod
    def _compute_motion_score_github(sing_values, threshold=0.99):
        sing_values_2 = np.square(sing_values)

        # compute Frobenius norm
        A_norm = np.sum(sing_values_2)

        # compute relative rank-approximation quality
        # the higher r, the higher the delta_r
        delta_r = np.sqrt(np.array([
            np.sum(sing_values_2[0:r + 1]) / A_norm
            for r in range(len(sing_values_2))
        ]))

        # find highest rank approximation that leads to error <
        # threshold
        r = np.where(delta_r < threshold)[0][-1] + 1
        # r = np.where(delta_r > threshold)[0][0] + 1  # lowest

        # surrogate motion score
        # NOTE: that's weird to me, should be the lower the better,
        # but delta_r refers to relative rank-approximation quality,
        # thus, the higher the better!
        motion_score = r * delta_r[r - 1]

        return motion_score

    ##
    # Use stack with largest mask volume as target stack
    # \date       2018-01-26 16:52:39+0000
    #
    # \param      cls               The cls
    # \param      file_paths_masks  paths to image masks as list of strings
    #
    @classmethod
    def from_volume(cls, file_paths_masks):
        t0 = ph.start_timing()
        target_stack_estimator = cls()

        volumes = np.array([
            TargetStackEstimator._compute_volume(f) for f in file_paths_masks
        ])

        # find index to smallest "valid" volume, i.e. volume > q * median
        index = np.argmax(
            volumes[np.argsort(volumes)] > 0.7 * np.median(volumes))
        index = np.argsort(volumes)[index]

        # Get index corresponding to maximum volume stack mask
        # index = np.argmax(volumes)
        # index = np.argmin(volumes)

        # Get index corresponding to median volume stack mask
        # index = np.argsort(volumes)[len(volumes)//2]

        target_stack_estimator._target_stack_index = index

        # computational time
        target_stack_estimator._computational_time = ph.stop_timing(t0)

        return target_stack_estimator

    ##
    # Compute target stack based on method presented in TMI Kainz2015. However,
    # only on masked anatomy (bounding box) is used in addition to a relative
    # rank weighting.
    # @date       2019-08-30 14:33:24+0100
    #
    # @param      cls               The cls
    # @param      file_paths        The file paths
    # @param      file_paths_masks  The file paths masks
    #
    # @return     target_stack_estimator instance
    #
    @classmethod
    def from_motion_score(cls, file_paths, file_paths_masks):

        if len(file_paths) != len(file_paths_masks):
            raise ValueError(
                "Number of provided images and masks must match")

        t0 = ph.start_timing()
        tse = cls()

        volumes = np.array([
            TargetStackEstimator._compute_volume(f) for f in file_paths_masks
        ])

        # only allow stacks with minimum volume, i.e. anatomical/brain coverage
        vol_min = 0.7 * np.median(volumes)
        indices = [i for i in range(len(volumes)) if volumes[i] > vol_min]

        # read all eligible stacks
        stacks = [
            st.Stack.from_filename(
                file_path=file_paths[i],
                file_path_mask=file_paths_masks[i],
                extract_slices=False,
            ) for i in indices
        ]

        # crop stack to bounding box of mask
        stacks = [s.get_cropped_stack_based_on_mask() for s in stacks]

        # debug
        # for i_stack, stack in enumerate(stacks):
        #     stack.show(label=str(indices[i_stack]))

        # compute motion scores of eligible stacks
        motion_scores = [None] * len(indices)
        for i_stack in range(len(indices)):

            # compute singular values
            s = TargetStackEstimator._compute_singular_values(stacks[i_stack])

            # compute motion score
            motion_scores[i_stack] = tse._compute_motion_score[tse._mode](s)

        # select stack with minimum motion (score)
        selection_best = np.argmin(motion_scores)

        # reference back to input file_paths index
        target_stack_index = indices[selection_best]

        tse._target_stack_index = target_stack_index

        # computational time
        tse._computational_time = ph.stop_timing(t0)

        # debug
        # print(indices, len(indices), len(file_paths))
        # print("Best: %d" % target_stack_index)
        # print(motion_scores)
        # print(tse.get_computational_time())
        # ph.killall_itksnap()

        return tse