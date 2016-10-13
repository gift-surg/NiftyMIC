## \file RegistrationITK.py
#  \brief This class makes ITK registration accessible via Python
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date June 2016

## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
import SimpleITKHelper as sitkh
import PSF as psf
import Stack as st

class RegistrationITK:

    def __init__(self, fixed=None, moving=None, use_fixed_mask=False, use_moving_mask=False, registration_type="Rigid", interpolator="NearestNeighbor", metric="Correlation", scales_estimator="Jacobian", ANTSradius=20, translation_scale=1, use_multiresolution_framework=False, dir_tmp="/tmp/RegistrationITK/", verbose=False):

        self._fixed = fixed
        self._moving = moving

        self._use_fixed_mask = use_fixed_mask
        self._use_moving_mask = use_moving_mask

        self._registration_type = registration_type
        self._metric = metric
        self._interpolator = interpolator
        self._scales_estimator = scales_estimator
        self._ANTSradius = ANTSradius
        self._translation_scale = translation_scale

        self._use_multiresolution_framework = use_multiresolution_framework

        self._use_verbose = verbose

        ## Temporary output where files are written in order to use ITK from the commandline
        self._dir_tmp = dir_tmp
        os.system("mkdir -p " + self._dir_tmp)
        os.system("rm -rf " + self._dir_tmp + "*")


        self._run_registration = {
            "Rigid"             : self._run_registration_rigid_affine,
            "Affine"            : self._run_registration_rigid_affine,
            "InplaneSimilarity" : self._run_registration_inplane_similarity
        }
        # self._transform_sitk = transform_sitk
        # self._control_point_grid_sitk = control_point_grid_sitk
        # self._registered_image = registered_image


    ## Set fixed/reference/target image
    #  \param[in] fixed fixed/reference/target image as Stack object
    def set_fixed(self, fixed):
        self._fixed = fixed


    ## Set moving/floating/source image
    #  \param[in] moving moving/floating/source image as Stack object
    def set_moving(self, moving):
        self._moving = moving


    ## Specify whether mask shall be used for fixed image
    #  \param[in] flag boolean
    def use_fixed_mask(self, flag):
        self._use_fixed_mask = flag


    ## Specify whether mask shall be used for moving image
    #  \param[in] flag boolean
    def use_moving_mask(self, flag):
        self._use_moving_mask = flag
        

    ## Set type of registration used
    #  \param[in] registration_type
    def set_registration_type(self, registration_type):
        if registration_type not in ["Rigid", "Affine", "InplaneSimilarity"]:
            raise ValueError("Error: Registration type can only be either 'Rigid', 'Affine' or 'InplaneSimilarity'")
        
        self._registration_type = registration_type


    ## Get chosen type of registration used
    #  \return registration type as string
    def get_registration_type(self):
        return self._registration_type


    ## Set interpolator
    #  \param[in] interpolator_type
    def set_interpolator(self, interpolator_type):
        if interpolator_type not in ["NearestNeighbor", "Linear", "BSpline", "OrientedGaussian"]:
            raise ValueError("Error: Interpolator can only be either 'NearestNeighbor', 'Linear', 'BSpline' or 'OrientedGaussian'")

        self._interpolator = interpolator_type


    ## Get interpolator
    #  \return interpolator as string
    def get_interpolator(self):
        return self._interpolator


    ## Set metric
    #  \param[in] metric
    def set_metric(self, metric):
        if metric not in ["MeanSquares", "MattesMutualInformation", "Correlation", "ANTSNeighborhoodCorrelation"]:
            raise ValueError("Error: Metric cannot be deduced.")

        self._metric = metric


    ## Get metric
    #  \return metric
    def get_metric(self):
        return self._metric


    ## Set/Get radius used for ANTSNeighborhoodCorrelation
    def set_ANTSradius(self, radius):
        self._ANTSradius = radius


    def get_ANTSradius(self):
        return self._ANTSradius


    ## Set/Get translation scale used for itkScaledTranslationEuler3DTransform
    def set_translation_scale(self, translation_scale):
        self._translation_scale = translation_scale

    def get_translation_scale(self):
        return self._translation_scale


    ## Decide whether multi-registration framework is used
    #  \param[in] flag
    def use_multiresolution_framework(self, flag):
        self._use_multiresolution_framework = flag


    ## Set scales estimator for optimizer
    #  \param[in] scales_estimator
    def set_scales_estimator(self, scales_estimator):
        if scales_estimator not in ["IndexShift", "PhysicalShift", "Jacobian"]:
            raise ValueError("Error: Metric cannot be deduced.")
        self._scales_estimator = scales_estimator


    ## Get scales estimator
    def get_scales_estimator(self):
        return self._scales_estimator


    ## Get affine transform in (Simple)ITK format after having run reg_aladin
    #  \return affine transform as SimpleITK object
    def get_registration_transform_sitk(self):
        return self._transform_sitk

    ##-------------------------------------------------------------------------
    # \brief      Gets the parameters obtained by the registration.
    # \date       2016-09-22 21:17:09+0100
    #
    # \param      self  The object
    #
    # \return     The parameters as numpy array.
    #
    def get_parameters(self):
        return np.array(self._parameters)


    ##-------------------------------------------------------------------------
    # \brief      Gets the fixed parameters obtained by the registration.
    # \date       2016-09-22 21:17:26+0100
    #
    # \param      self  The object
    #
    # \return     The fixed parameters as numpy array.
    #
    def get_parameters_fixed(self):
        return np.array(self._parameters_fixed)


    ## Get registered image
    #  \return registered image as Stack object
    def get_registered_image(self):
        return self._registered_image


    ##-------------------------------------------------------------------------
    # \brief      Sets the verbose to define whether or not output is produced
    # \date       2016-09-20 18:49:19+0100
    #
    # \param      self     The object
    # \param      verbose  The verbose
    #
    def use_verbose(self, flag):
        self._use_verbose = flag


    ##-------------------------------------------------------------------------
    # \brief      Gets the verbose.
    # \date       2016-09-20 18:49:54+0100
    #
    # \param      self  The object
    #
    # \return     The verbose.
    #
    def get_verbose(self):
        return self._use_verbose


    def run_registration(self, id=""):
        self._run_registration[self._registration_type](id)


    def _run_registration_rigid_affine(self, id):

        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

        if self._use_verbose:
            verbose = "1"
        else:
            verbose = "0"

        ## Clean output directory first
        os.system("rm -rf " + self._dir_tmp + "*")

        moving_str = "RegistrationITK_moving_" + id + self._moving.get_filename()
        fixed_str = "RegistrationITK_fixed_" + id + self._fixed.get_filename()
        moving_mask_str = "RegistrationITK_moving_mask_" + id + self._moving.get_filename() 
        fixed_mask_str = "RegistrationITK_fixed_mask_" + id + self._fixed.get_filename()
        
        registration_transform_str = "RegistrationITK_transform_" + id + self._fixed.get_filename() + "_" + self._moving.get_filename()

        ## Write images to HDD
        # if not os.path.isfile(self._dir_tmp + moving_str + ".nii.gz"):
        sitk.WriteImage(self._moving.sitk, self._dir_tmp + moving_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + fixed_str + ".nii.gz"):
        sitk.WriteImage(self._fixed.sitk, self._dir_tmp + fixed_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + moving_mask_str + ".nii.gz") and self._use_moving_mask:
        sitk.WriteImage(self._moving.sitk_mask, self._dir_tmp + moving_mask_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + fixed_mask_str + ".nii.gz") and self._use_fixed_mask:
        sitk.WriteImage(self._fixed.sitk_mask, self._dir_tmp + fixed_mask_str + ".nii.gz")

        ## Prepare command for execution
        cmd =  "/Users/mebner/UCL/UCL/Volumetric\ Reconstruction/cpp/build/bin/itkReg "
        cmd += "--f " + self._dir_tmp + fixed_str + ".nii.gz "
        cmd += "--m " + self._dir_tmp + moving_str + ".nii.gz "
        if self._use_fixed_mask:
            cmd += "--fmask " + self._dir_tmp + fixed_mask_str + ".nii.gz "
        if self._use_moving_mask:
            cmd += "--mmask " + self._dir_tmp + moving_mask_str + ".nii.gz "
        cmd += "--tout " + self._dir_tmp + registration_transform_str + ".txt "
        cmd += "--useAffine " + str(int(self._registration_type is "Affine")) + " "
        cmd += "--useMultires " + str(int(self._use_multiresolution_framework)) + " "
        cmd += "--metric " + self._metric + " "
        cmd += "--scalesEst " + self._scales_estimator + " "
        cmd += "--interpolator " + self._interpolator + " "
        cmd += "--ANTSrad " + str(self._ANTSradius) + " "
        cmd += "--translationScale " + str(self._translation_scale) + " "
        cmd += "--verbose " + verbose + " "

        ## Compute oriented Gaussian PSF if desired
        if self._interpolator in ["OrientedGaussian"]:
            ## Get oriented Gaussian covariance matrix
            cov_HR_coord = psf.PSF().get_gaussian_PSF_covariance_matrix_HR_volume_coordinates( self._fixed, self._moving ).flatten()
            cmd += "--cov " + "'" + ' '.join(cov_HR_coord.astype("|S12")) + "'"

        # if self._use_verbose:
        print("\t----- print command -----------------------------")
        print cmd
        print("\t-------------------------------------------------")
        os.system(cmd)

        ## Read transformation file
        params_all = np.loadtxt(self._dir_tmp + registration_transform_str + ".txt")

        self._parameters_fixed = params_all[0:3]
        self._parameters = params_all[3:]

        if self._registration_type in ["Rigid"]:
            self._transform_sitk = sitk.Euler3DTransform()
        else:
            self._transform_sitk = sitk.AffineTransform(3)

        self._transform_sitk.SetParameters(self._parameters)
        self._transform_sitk.SetFixedParameters(self._parameters_fixed)

        ## Debug
        # moving_warped_sitk = sitk.Resample(self._moving.sitk, self._fixed.sitk, self._transform_sitk, sitk.sitkLinear, 0.0, self._moving.sitk.GetPixelIDValue())
        # sitk.WriteImage(moving_warped_sitk, self._dir_tmp + "RegistrationITK_result.nii.gz")


    def _run_registration_inplane_similarity(self, id):

        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

        if self._use_verbose:
            verbose = "1"
        else:
            verbose = "0"

        ## Clean output directory first
        os.system("rm -rf " + self._dir_tmp + "*")

        moving_str = "RegistrationITK_moving_" + id + self._moving.get_filename()
        fixed_str = "RegistrationITK_fixed_" + id + self._fixed.get_filename()
        moving_mask_str = "RegistrationITK_moving_mask_" + id + self._moving.get_filename() 
        fixed_mask_str = "RegistrationITK_fixed_mask_" + id + self._fixed.get_filename()
        
        registration_transform_str = "RegistrationITK_transform_" + id + self._fixed.get_filename() + "_" + self._moving.get_filename()

        ## Write images to HDD
        # if not os.path.isfile(self._dir_tmp + moving_str + ".nii.gz"):
        sitk.WriteImage(self._moving.sitk, self._dir_tmp + moving_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + fixed_str + ".nii.gz"):
        sitk.WriteImage(self._fixed.sitk, self._dir_tmp + fixed_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + moving_mask_str + ".nii.gz") and self._use_moving_mask:
        sitk.WriteImage(self._moving.sitk_mask, self._dir_tmp + moving_mask_str + ".nii.gz")
        # if not os.path.isfile(self._dir_tmp + fixed_mask_str + ".nii.gz") and self._use_fixed_mask:
        sitk.WriteImage(self._fixed.sitk_mask, self._dir_tmp + fixed_mask_str + ".nii.gz")

        ## Prepare command for execution
        cmd =  "/Users/mebner/UCL/UCL/Volumetric\ Reconstruction/cpp/build/bin/itkInplaneSimilarity3DReg "
        cmd += "--f " + self._dir_tmp + fixed_str + ".nii.gz "
        cmd += "--m " + self._dir_tmp + moving_str + ".nii.gz "
        if self._use_fixed_mask:
            cmd += "--fmask " + self._dir_tmp + fixed_mask_str + ".nii.gz "
        if self._use_moving_mask:
            cmd += "--mmask " + self._dir_tmp + moving_mask_str + ".nii.gz "
        cmd += "--tout " + self._dir_tmp + registration_transform_str + ".txt "
        cmd += "--useAffine " + str(int(self._registration_type is "Affine")) + " "
        cmd += "--useMultires " + str(int(self._use_multiresolution_framework)) + " "
        cmd += "--metric " + self._metric + " "
        cmd += "--scalesEst " + self._scales_estimator + " "
        cmd += "--interpolator " + self._interpolator + " "
        cmd += "--ANTSrad " + str(self._ANTSradius) + " "
        cmd += "--translationScale " + str(self._translation_scale) + " "
        cmd += "--verbose " + verbose + " "

        ## Compute oriented Gaussian PSF if desired
        if self._interpolator in ["OrientedGaussian"]:
            ## Get oriented Gaussian covariance matrix
            cov_HR_coord = psf.PSF().get_gaussian_PSF_covariance_matrix_HR_volume_coordinates( self._fixed, self._moving ).flatten()
            cmd += "--cov " + "'" + ' '.join(cov_HR_coord.astype("|S12")) + "'"

        # if self._use_verbose:
        print("\t----- print command -----------------------------")
        print cmd
        print("\t-------------------------------------------------")
        os.system(cmd)

        ## Read transformation file
        params_all = np.loadtxt(self._dir_tmp + registration_transform_str + ".txt")

        self._parameters_fixed = params_all[0:-7]
        self._parameters = params_all[-7:]


        # transform_sitk = sitk.Euler3DTransform()
        # transform_sitk.SetParameters(parameters[0:-1])
        # transform_sitk.SetFixedParameters(parameters_fixed[0:3])

        # ## Debug
        # scale = parameters[-1]
        # spacing = np.array(self._moving.sitk.GetSpacing())
        # spacing[0:-1] *= scale
        # moving_sitk = sitk.Image(self._moving.sitk)
        # moving_sitk.SetSpacing(spacing)
        # moving_warped_sitk = sitk.Resample(moving_sitk, self._fixed.sitk, transform_sitk, sitk.sitkLinear)
        # sitk.WriteImage(moving_warped_sitk, self._dir_tmp + "RegistrationITK_result.nii.gz")

        # sitkh.show_sitk_image([self._fixed.sitk, moving_warped_sitk], ["fixed", "moving_registered"])
