##
# \file cpp_itk_registration.py
# \brief      This class makes ITK registration accessible via Python
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       June 2016
#

# Import libraries
import os
import sys
import SimpleITK as sitk
import numpy as np


# Import modules from src-folder
import pysitk.simple_itk_helper as sitkh
import pysitk.python_helper as ph

import niftymic.base.psf as psf
import niftymic.base.stack as st
import niftymic.base.slice as sl

from niftymic.registration.simple_itk_registration \
    import SimpleItkRegistration
from niftymic.definitions import DIR_TMP
from niftymic.definitions import DIR_CPP_BUILD


class CppItkRegistration(SimpleItkRegistration):

    ##
    # { constructor_description }
    # \date       2017-08-12 13:51:58+0100
    #
    # \param      self                           The object
    # \param      fixed                          The fixed
    # \param      moving                         The moving
    # \param      use_fixed_mask                 The use fixed mask
    # \param      use_moving_mask                The use moving mask
    # \param      registration_type              The registration type
    # \param      interpolator                   ["Linear", OrientedGaussian"]
    # \param      metric                         The metric
    # \param      scales_estimator               The scales estimator
    # \param      use_multiresolution_framework  The use multiresolution framework
    # \param      use_verbose                    The use verbose
    # \param      ANTSradius                     The ant sradius
    # \param      translation_scale              The translation scale
    # \param      dir_tmp                        The dir temporary
    #
    def __init__(self,
                 fixed=None,
                 moving=None,
                 cov=None,
                 use_fixed_mask=False,
                 use_moving_mask=False,
                 registration_type="Rigid",
                 interpolator="Linear",
                 metric="Correlation",
                 scales_estimator="PhysicalShift",
                 use_multiresolution_framework=False,
                 use_verbose=False,
                 ANTSradius=20,
                 dir_tmp=os.path.join(DIR_TMP, "CppItkRegistration"),
                 ):

        SimpleItkRegistration.__init__(
            self,
            fixed=fixed,
            moving=moving,
            use_fixed_mask=use_fixed_mask,
            use_moving_mask=use_moving_mask,
            registration_type=registration_type,
            interpolator=interpolator,
            metric=metric,
            metric_params=None,
            optimizer=None,
            optimizer_params=None,
            scales_estimator=scales_estimator,
            initializer_type=None,
            use_oriented_psf=None,
            use_multiresolution_framework=use_multiresolution_framework,
            shrink_factors=None,
            smoothing_sigmas=None,
            use_verbose=use_verbose,
        )
        self._cov = cov
        self._REGISTRATION_TYPES = ["Rigid", "Affine", "InplaneSimilarity"]
        self._INITIALIZER_TYPES = [None]
        self._SCALES_ESTIMATORS = ["IndexShift", "PhysicalShift", "Jacobian"]

        self._ANTSradius = ANTSradius

        self._use_verbose = use_verbose

        # Temporary output where files are written in order to use ITK from the
        # commandline
        self._dir_tmp = ph.create_directory(dir_tmp, delete_files=False)

        self._run_registration_ = {
            "Rigid": self._run_registration_rigid_affine,
            "Affine": self._run_registration_rigid_affine,
            "InplaneSimilarity": self._run_registration_inplane_similarity_3D
        }

    # Set/Get radius used for ANTSNeighborhoodCorrelation
    def set_ANTSradius(self, radius):
        self._ANTSradius = radius

    def get_ANTSradius(self):
        return self._ANTSradius

    ##
    #       Gets the parameters obtained by the registration.
    # \date       2016-09-22 21:17:09+0100
    #
    # \param      self  The object
    #
    # \return     The parameters as numpy array.
    #
    def get_parameters(self):
        return np.array(self._parameters)

    ##
    #       Gets the fixed parameters obtained by the registration.
    # \date       2016-09-22 21:17:26+0100
    #
    # \param      self  The object
    #
    # \return     The fixed parameters as numpy array.
    #
    def get_parameters_fixed(self):
        return np.array(self._parameters_fixed)

    # Get registered image
    #  \return registered image as Stack object
    # def get_transformed_fixed(self):
        # return self._registered_image

    def _run(self, id=""):

        # Clean output directory first
        ph.clear_directory(self._dir_tmp, verbose=0)

        self._run_registration_[self._registration_type](id)

    def _run_registration_rigid_affine(self, id, endl=" \\\n"):

        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

        if self._use_verbose:
            verbose = "1"
        else:
            verbose = "0"

        moving_str = "RegistrationITK_moving_" + id + self._moving.get_filename()
        fixed_str = "RegistrationITK_fixed_" + id + self._fixed.get_filename()
        moving_mask_str = "RegistrationITK_moving_mask_" + id + self._moving.get_filename()
        fixed_mask_str = "RegistrationITK_fixed_mask_" + id + self._fixed.get_filename()

        registration_transform_str = "RegistrationITK_transform_" + id + \
            self._fixed.get_filename() + "_" + self._moving.get_filename()

        # Write images to HDD
        # if not os.path.isfile(self._dir_tmp + moving_str + ".nii.gz"):
        sitkh.write_nifti_image_sitk(self._moving.sitk, self._dir_tmp +
                                     moving_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + fixed_str + ".nii.gz"):
        sitkh.write_nifti_image_sitk(self._fixed.sitk, self._dir_tmp +
                                     fixed_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + moving_mask_str + ".nii.gz")
        # and self._use_moving_mask:
        sitkh.write_nifti_image_sitk(self._moving.sitk_mask,
                                     self._dir_tmp + moving_mask_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + fixed_mask_str + ".nii.gz") and
        # self._use_fixed_mask:
        sitkh.write_nifti_image_sitk(self._fixed.sitk_mask, self._dir_tmp +
                                     fixed_mask_str + ".nii.gz")

        # Prepare command for execution
        # cmd =  "/Users/mebner/UCL/UCL/Software/Volumetric\ Reconstruction/build/cpp/bin/itkReg "
        cmd = DIR_CPP_BUILD + "/bin/itkReg" + endl
        cmd += "--f " + self._dir_tmp + fixed_str + ".nii.gz" + endl
        cmd += "--m " + self._dir_tmp + moving_str + ".nii.gz" + endl
        if self._use_fixed_mask:
            cmd += "--fmask " + self._dir_tmp + fixed_mask_str + ".nii.gz" + endl
        if self._use_moving_mask:
            cmd += "--mmask " + self._dir_tmp + moving_mask_str + ".nii.gz" + endl
        cmd += "--tout " + self._dir_tmp + registration_transform_str + ".txt" + endl
        cmd += "--useAffine " + \
            str(int(self._registration_type is "Affine")) + endl
        cmd += "--useMultires " + \
            str(int(self._use_multiresolution_framework)) + endl
        cmd += "--metric " + self._metric + endl
        cmd += "--scalesEst " + self._scales_estimator + endl
        cmd += "--interpolator " + self._interpolator + endl
        cmd += "--ANTSrad " + str(self._ANTSradius) + endl
        cmd += "--verbose " + verbose + endl

        # Compute oriented Gaussian PSF if desired
        if self._interpolator in ["OrientedGaussian"]:
            if self._cov is None:
                # Get oriented Gaussian covariance matrix
                cov_HR_coord = psf.PSF().\
                    get_covariance_matrix_in_reconstruction_space(
                    self._moving, self._fixed).flatten()
            else:
                cov_HR_coord = self._cov.flatten()
            cmd += "--cov " + "'" + ' '.join(cov_HR_coord.astype("|S12")) + "'"

        ph.execute_command(cmd, verbose=0)

        # Read transformation file
        params_all = np.loadtxt(
            self._dir_tmp + registration_transform_str + ".txt")

        if self._registration_type in ["Rigid"]:
            self._parameters_fixed = params_all[0:4]
            self._parameters = params_all[4:]
            self._registration_transform_sitk = sitk.Euler3DTransform()

        else:
            self._parameters_fixed = params_all[0:3]
            self._parameters = params_all[3:]
            self._registration_transform_sitk = sitk.AffineTransform(3)

        self._registration_transform_sitk.SetParameters(self._parameters)
        self._registration_transform_sitk.SetFixedParameters(
            self._parameters_fixed)

        # Debug
        # moving_warped_sitk = sitk.Resample(self._moving.sitk, self._fixed.sitk, self._registration_transform_sitk, sitk.sitkLinear, 0.0, self._moving.sitk.GetPixelIDValue())
        # sitkh.write_nifti_image_sitk(moving_warped_sitk, self._dir_tmp + "RegistrationITK_result.nii.gz")

    def _run_registration_inplane_similarity_3D(self, id, endl=" \\\n"):

        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

        if self._use_verbose:
            verbose = "1"
        else:
            verbose = "0"

        # Clean output directory first
        os.system("rm -rf " + self._dir_tmp + "*")

        moving_str = "RegistrationITK_moving_" + id + self._moving.get_filename()
        fixed_str = "RegistrationITK_fixed_" + id + self._fixed.get_filename()
        moving_mask_str = "RegistrationITK_moving_mask_" + id + self._moving.get_filename()
        fixed_mask_str = "RegistrationITK_fixed_mask_" + id + self._fixed.get_filename()

        registration_transform_str = "RegistrationITK_transform_" + id + \
            self._fixed.get_filename() + "_" + self._moving.get_filename()

        # Write images to HDD
        # if not os.path.isfile(self._dir_tmp + moving_str + ".nii.gz"):
        sitkh.write_nifti_image_sitk(self._moving.sitk, self._dir_tmp +
                                     moving_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + fixed_str + ".nii.gz"):
        sitkh.write_nifti_image_sitk(self._fixed.sitk, self._dir_tmp +
                                     fixed_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + moving_mask_str + ".nii.gz")
        # and self._use_moving_mask:
        sitkh.write_nifti_image_sitk(self._moving.sitk_mask,
                                     self._dir_tmp + moving_mask_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + fixed_mask_str + ".nii.gz") and
        # self._use_fixed_mask:
        sitkh.write_nifti_image_sitk(self._fixed.sitk_mask, self._dir_tmp +
                                     fixed_mask_str + ".nii.gz")

        # Prepare command for execution
        cmd = DIR_CPP_BUILD + "/bin/itkInplaneSimilarity3DReg" + endl
        cmd += "--f " + self._dir_tmp + fixed_str + ".nii.gz" + endl
        cmd += "--m " + self._dir_tmp + moving_str + ".nii.gz" + endl
        if self._use_fixed_mask:
            cmd += "--fmask " + self._dir_tmp + fixed_mask_str + ".nii.gz" + endl
        if self._use_moving_mask:
            cmd += "--mmask " + self._dir_tmp + moving_mask_str + ".nii.gz" + endl
        cmd += "--tout " + self._dir_tmp + registration_transform_str + ".txt" + endl
        cmd += "--useAffine " + \
            str(int(self._registration_type is "Affine")) + endl
        cmd += "--useMultires " + \
            str(int(self._use_multiresolution_framework)) + endl
        cmd += "--metric " + self._metric + endl
        cmd += "--scalesEst " + self._scales_estimator + endl
        cmd += "--interpolator " + self._interpolator + endl
        cmd += "--ANTSrad " + str(self._ANTSradius) + endl
        # cmd += "--translationScale " + str(self._translation_scale) + endl
        cmd += "--verbose " + verbose + endl

        # Compute oriented Gaussian PSF if desired
        if self._interpolator in ["OrientedGaussian"]:
            if self._cov is None:
                # Get oriented Gaussian covariance matrix
                cov_HR_coord = psf.PSF().\
                    get_covariance_matrix_in_reconstruction_space(
                    self._moving, self._fixed).flatten()
            else:
                cov_HR_coord = self._cov.flatten()
            cmd += "--cov " + "'" + ' '.join(cov_HR_coord.astype("|S12")) + "'"

        # if self._use_verbose:
        ph.execute_command(cmd)

        # Read transformation file
        params_all = np.loadtxt(
            self._dir_tmp + registration_transform_str + ".txt")

        ## (center_x, center_y, center_z, direction_fixed_image_flattened_0, ..., direction_fixed_image_flattened_8)
        self._parameters_fixed = params_all[0:-7]

        ## (versor_0, versor_1, versor_2, translation_x, translation_y, translation_z, scale)
        self._parameters = params_all[-7:]

        # Get affine registration transform T(x) = R D Lambda D^{-1} (x-c) + t
        # + c
        self._registration_transform_sitk = self._get_affine_transform_from_similarity_registration()

        # ## Debug
        # scale = parameters[-1]
        # spacing = np.array(self._moving.sitk.GetSpacing())
        # spacing[0:-1] *= scale
        # moving_sitk = sitk.Image(self._moving.sitk)
        # moving_sitk.SetSpacing(spacing)
        # moving_warped_sitk = sitk.Resample(moving_sitk, self._fixed.sitk, transform_sitk, sitk.sitkLinear)
        # sitkh.write_nifti_image_sitk(moving_warped_sitk, self._dir_tmp + "RegistrationITK_result.nii.gz")

        # sitkh.show_sitk_image([self._fixed.sitk, moving_warped_sitk], ["fixed", "moving_registered"])
        #

    ##
    #       Gets the affine transform from similarity registration.
    # \date       2016-11-02 15:55:00+0000
    #
    #  Returns registration transform
    #  \f[ T(\vec{x})
    #    = R\,D\,\Lambda\,D^{-1} (\vec{x} - \vec{c}) + \vec{t} + \vec{c} \f]
    #  which corresponds to the implemented similarity 3D transform. Matrix D
    #  corresponds to direction matrix of the fixed image, matrix R the
    #  rotation matrix, Lambda = diag(s,s,1) the scaling matrix, c the center
    #  and t the translation.
    #
    # \param      self  The object
    #
    # \return     The affine transform from similarity registration.
    #
    def _get_affine_transform_from_similarity_registration(self):

        # Extract information from (fixed) parameters
        center = self._parameters_fixed[0:3]
        versor = self._parameters[0:3]
        translation = self._parameters[3:6]
        scale = self._parameters[6]

        # Extract information from image
        spacing = np.array(self._fixed.sitk.GetSpacing())
        origin = np.array(self._fixed.sitk.GetOrigin())
        D = np.array(self._fixed.sitk.GetDirection()).reshape(3, 3)
        D_inv = np.linalg.inv(D)

        # Create scaling matrix Lambda
        Lambda = np.eye(3)
        Lambda[0:-1, :] *= scale

        # Get rotation matrix from parameters
        rigid_sitk = sitk.VersorRigid3DTransform()
        rigid_sitk.SetParameters(self._parameters[0:6])
        R = np.array(rigid_sitk.GetMatrix()).reshape(3, 3)

        # Create Affine Transform based on given registration transform
        affine_transform_sitk = sitk.AffineTransform(3)
        affine_transform_sitk.SetMatrix(
            (R.dot(D).dot(Lambda).dot(D_inv)).flatten())
        affine_transform_sitk.SetCenter(center)
        affine_transform_sitk.SetTranslation(translation)

        return affine_transform_sitk

    ##
    #       Gets the rigid transform and scale from similarity registration.
    # \date       2016-11-02 15:55:57+0000
    #
    # Returns rigid transform (center set to zero)
    # \f$ T(\vec{x})] =  R\,\vec{x} + \vec{t\prime} \f$
    # with \f$ \vec{t} = R\,D\,\Lambda\,D^{-1}(\vec{o}-\vec{c}) - R\vec{o} + \vec{t} + \vec{c} \f$
    # with o being the reference image origin, and t and c the respective
    # parameters from the affine transform (i.e. similarity registration
    # transform)
    #
    # \param      self  The object
    #
    # \return     The rigid transform and scale from similarity registration.
    #
    def _get_rigid_transform_and_scaling_from_similarity_registration(self):

        # Extract information from (fixed) parameters
        center = self._parameters_fixed[0:3]
        versor = self._parameters[0:3]
        translation = self._parameters[3:6]
        scale = self._parameters[6]

        # Extract information from image
        spacing = np.array(self._fixed.sitk.GetSpacing())
        origin = np.array(self._fixed.sitk.GetOrigin())
        D = np.array(self._fixed.sitk.GetDirection()).reshape(3, 3)
        D_inv = np.linalg.inv(D)

        # Create scaling matrix Lambda
        Lambda = np.eye(3)
        Lambda[0:-1, :] *= scale

        # Create Rigid Transform based on given registration information
        rigid_sitk = sitk.VersorRigid3DTransform()
        rigid_sitk.SetParameters(self._parameters[0:6])
        R = np.array(rigid_sitk.GetMatrix()).reshape(3, 3)
        rigid_sitk.SetTranslation(R.dot(D).dot(Lambda).dot(D_inv).dot(
            origin - center) - R.dot(origin) + translation + center)

        return rigid_sitk, scale

    ##
    #       Gets the stack with similarity inplane transformed slices.
    # \date       2016-11-02 18:21:17+0000
    #
    # \param      self           The object
    # \param      stack_to_copy  Stack as Stack object
    #
    # \return     The stack with similarity inplane transformed slices
    # according to preceding similarity registration
    #
    def get_stack_with_similarity_inplane_transformed_slices(self, stack_to_copy):

        stack = st.Stack.from_stack(
            stack_to_copy, filename=stack_to_copy.get_filename())

        # Get
        rigid_sitk, scale = self._get_rigid_transform_and_scaling_from_similarity_registration()

        # Extract information from image
        spacing_scaled = np.array(self._fixed.sitk.GetSpacing())
        spacing_scaled[0:-1] *= scale

        # Create 3D image based on the obtained (in-plane) similarity transform
        stack_inplaneSimilar_sitk = sitkh.get_transformed_sitk_image(
            self._fixed.sitk, rigid_sitk)
        stack_inplaneSimilar_sitk_mask = sitkh.get_transformed_sitk_image(
            self._fixed.sitk_mask, rigid_sitk)
        stack_inplaneSimilar_sitk.SetSpacing(spacing_scaled)
        stack_inplaneSimilar_sitk_mask.SetSpacing(spacing_scaled)
        stack_inplaneSimilar = st.Stack.from_sitk_image(
            image_sitk=stack_inplaneSimilar_sitk,
            filename=stack.get_filename(),
            image_sitk_mask=stack_inplaneSimilar_sitk_mask,
            slice_thickness=stack.get_slice_thickness(),
        )

        # Update all its slices based on the obtained (in-plane) similarity
        # transform
        slices_stack_inplaneSimilar = stack_inplaneSimilar.get_slices()
        slices_stack = stack.get_slices()

        N_slices = stack_inplaneSimilar.get_number_of_slices()

        for i in range(0, N_slices):
            slice_sitk = slices_stack[i].sitk
            slice_sitk_mask = slices_stack[i].sitk_mask

            slice_sitk.SetSpacing(spacing_scaled)
            slice_sitk_mask.SetSpacing(spacing_scaled)

            slice = sl.Slice.from_sitk_image(
                slice_sitk=slice_sitk,
                filename=slices_stack[i].get_filename(),
                slice_number=slices_stack[i].get_slice_number(),
                slice_sitk_mask=slice_sitk_mask,
                slice_thickness=slices_stack[i].get_slice_thickness(),
            )
            slice.update_motion_correction(rigid_sitk)

            slices_stack_inplaneSimilar[i] = slice

        return stack_inplaneSimilar
