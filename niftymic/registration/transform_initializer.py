# \file transform_initializer.py
# \brief      Class to obtain transform estimate to align fixed with moving
#             image
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Feb 2019
#

import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
import nsol.principal_component_analysis as pca
from nsol.similarity_measures import SimilarityMeasures

import niftymic.base.stack as st
import niftymic.validation.image_similarity_evaluator as ise
import niftymic.utilities.template_stack_estimator as tse

from niftymic.definitions import DIR_TMP


##
# Class to obtain transform estimate to align fixed with moving image
# \date       2019-02-20 17:47:54+0000
#
class TransformInitializer(object):

    def __init__(self,
                 fixed,
                 moving,
                 similarity_measure="NMI",
                 refine_pca_initializations=False,
                 ):
        self._fixed = fixed
        self._moving = moving
        self._similarity_measure = similarity_measure
        self._refine_pca_initializations = refine_pca_initializations

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

        if self._refine_pca_initializations:
            transformations = self._run_registrations(transformations)

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
            verbose=False,
        )
        ph.print_info(
            "Find best aligning transform as measured by %s" %
            self._similarity_measure)
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
            for i in range(len(transformations)):
                print("%s: %.6f" % (
                    labels[1 + i], similarities[self._similarity_measure][i])
                )

        return transform_init_sitk

    def _run_registrations(self, transformations):
        path_to_fixed = os.path.join(DIR_TMP, "fixed.nii.gz")
        path_to_moving = os.path.join(DIR_TMP, "moving.nii.gz")
        path_to_fixed_mask = os.path.join(DIR_TMP, "fixed_mask.nii.gz")
        path_to_moving_mask = os.path.join(DIR_TMP, "moving_mask.nii.gz")
        path_to_tmp_output = os.path.join(DIR_TMP, "foo.nii.gz")
        path_to_transform_regaladin = os.path.join(
            DIR_TMP, "transform_regaladin.txt")
        path_to_transform_sitk = os.path.join(
            DIR_TMP, "transform_sitk.txt")

        sitkh.write_nifti_image_sitk(self._fixed.sitk, path_to_fixed)
        sitkh.write_nifti_image_sitk(self._moving.sitk, path_to_moving)
        sitkh.write_nifti_image_sitk(self._fixed.sitk_mask, path_to_fixed_mask)
        # sitkh.write_nifti_image_sitk(
        #     self._moving.sitk_mask, path_to_moving_mask)

        for i in range(len(transformations)):
            sitk.WriteTransform(transformations[i], path_to_transform_sitk)

            # Convert SimpleITK to RegAladin transform
            cmd = "simplereg_transform -sitk2nreg %s %s" % (
                path_to_transform_sitk, path_to_transform_regaladin)
            ph.execute_command(cmd, verbose=False)

            # Run NiftyReg
            cmd_args = ["reg_aladin"]
            cmd_args.append("-ref %s" % path_to_fixed)
            cmd_args.append("-flo %s" % path_to_moving)
            cmd_args.append("-res %s" % path_to_tmp_output)
            cmd_args.append("-inaff %s" % path_to_transform_regaladin)
            cmd_args.append("-aff %s" % path_to_transform_regaladin)
            cmd_args.append("-rigOnly")
            cmd_args.append("-ln 2")
            cmd_args.append("-voff")
            cmd_args.append("-rmask %s" % path_to_fixed_mask)
            # To avoid error "0 correspondences between blocks were found" that can
            # occur for some cases. Also, disable moving mask, as this would be ignored
            # anyway
            cmd_args.append("-noSym")
            ph.print_info(
                "Run Registration based on PCA-init %d ... " % (i + 1))
            ph.execute_command(" ".join(cmd_args), verbose=False)

            # Convert RegAladin to SimpleITK transform
            cmd = "simplereg_transform -nreg2sitk %s %s" % (
                path_to_transform_regaladin, path_to_transform_sitk)
            ph.execute_command(cmd, verbose=False)

            transformations[i] = sitkh.read_transform_sitk(
                path_to_transform_sitk)

        return transformations
