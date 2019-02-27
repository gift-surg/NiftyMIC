# \file transform_initializer.py
# \brief      Class to obtain transform estimate to align fixed with moving
#             image
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Feb 2019
#

import numpy as np
import SimpleITK as sitk

import nsol.principal_component_analysis as pca
from nsol.similarity_measures import SimilarityMeasures

import niftymic.base.stack as st
import niftymic.validation.image_similarity_evaluator as ise
import niftymic.utilities.template_stack_estimator as tse


##
# Class to obtain transform estimate to align fixed with moving image
# \date       2019-02-20 17:47:54+0000
#
class TransformInitializer(object):

    def __init__(self, fixed, moving, similarity_measure="NMI"):
        self._fixed = fixed
        self._moving = moving
        self._similarity_measure = similarity_measure

        self._initial_transform_sitk = None

    def get_transform_sitk(self):
        return self._initial_transform_sitk

    def run(self, debug=False):
        # perform PCAs for fixed and moving images
        pca_moving = self.get_pca_from_mask(self._moving.sitk_mask)
        eigvec_moving = pca_moving.get_eigvec()
        mean_moving = pca_moving.get_mean()

        pca_fixed = self.get_pca_from_mask(self._fixed.sitk_mask)
        eigvec_fixed = pca_fixed.get_eigvec()
        mean_fixed = pca_fixed.get_mean()

        # test different initializations based on eigenvector orientations
        orientations = [
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ]
        transformations = []
        for i_o, orientation in enumerate(orientations):
            eigvec_moving_o = np.array(eigvec_moving)
            eigvec_moving_o[:, 0] *= orientation[0]
            eigvec_moving_o[:, 1] *= orientation[1]

            # get right-handed coordinate system
            cross = np.cross(eigvec_moving_o[:, 0], eigvec_moving_o[:, 1])
            eigvec_moving_o[:, 2] = cross

            # transformation to align fixed with moving eigenbasis
            R = eigvec_moving_o.dot(eigvec_fixed.transpose())
            t = mean_moving - R.dot(mean_fixed)

            # build rigid transformation as sitk object
            rigid_transform_sitk = sitk.Euler3DTransform()
            rigid_transform_sitk.SetMatrix(R.flatten())
            rigid_transform_sitk.SetTranslation(t)
            transformations.append(rigid_transform_sitk)

        # get best transformation according to selected similarity measure
        self._initial_transform_sitk = self._get_best_transform(
            transformations)

        if debug:
            foo = sitk.Resample(
                self._moving.sitk,
                self._fixed.sitk,
                self._initial_transform_sitk,
            )
            sitkh.show_sitk_image([fixed.sitk, foo])

    @staticmethod
    def get_pca_from_mask(mask_sitk, robust=False):
        mask_nda = sitk.GetArrayFromImage(mask_sitk)

        # get largest connected region (if more than one connected region)
        mask_nda = tse.TemplateStackEstimator.get_largest_connected_region_mask(
            mask_nda)

        # [z, y, x] x n_points to [x, y, z] x n_points
        points = np.array(np.where(mask_nda > 0))[::-1, :]
        n_points = len(points[0])
        for i in range(n_points):
            points[:, i] = mask_sitk.TransformIndexToPhysicalPoint(
                [int(j) for j in points[:, i]])

        if robust:
            pca_mask = pca.AdmmRobustPrincipalComponentAnalysis(
                points.transpose())
            res = pca_mask.run()

            pca_mask = pca.PrincipalComponentAnalysis(res["X3_admm"])
            pca_mask.run()

        else:
            pca_mask = pca.PrincipalComponentAnalysis(points.transpose())
            pca_mask.run()

        return pca_mask

    def _get_best_transform(self, transformations, debug=False):
        warps = []
        for transform_sitk in transformations:
            warped_moving_sitk = sitk.Resample(
                self._moving.sitk,
                self._fixed.sitk,
                transform_sitk,
                sitk.sitkLinear,
            )
            warps.append(
                st.Stack.from_sitk_image(
                    warped_moving_sitk,
                    extract_slices=False,
                    slice_thickness=self._fixed.get_slice_thickness(),
                ))

        image_similarity_evaluator = ise.ImageSimilarityEvaluator(
            stacks=warps,
            reference=self._fixed,
            measures=[self._similarity_measure],
            use_reference_mask=True,
        )
        image_similarity_evaluator.compute_similarities()
        similarities = image_similarity_evaluator.get_similarities()

        # get transform which leads to highest similarity
        index = np.argmax(similarities[self._similarity_measure])
        transform_init_sitk = transformations[index]

        if debug:
            labels = ["attempt%d" % (d + 1)
                      for d in range(len(transformations))]
            labels[index] = "best"
            foo = [w.sitk for w in warps]
            foo.insert(0, self._fixed.sitk)
            labels.insert(0, "fixed")
            sitkh.show_sitk_image(foo, label=labels)

        return transform_init_sitk
