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

    def __init__(self):
        self._moving = None
        self._fixed = None

        self._use_fixed_mask = False
        self._use_moving_mask = False

        self._transform_sitk = None
        self._control_point_grid_sitk = None
        self._registered_image = None

        self._metric = "Correlation"
        self._interpolator = "BSpline"
        self._scales_estimator = "Jacobian"
        self._ANTSradius = 20
        self._translation_scale = 1

        self._use_multiresolution_framework = 0

        ## Temporary output where files are written in order to use ITK from the commandline
        self._dir_tmp = "/tmp/RegistrationITK/"
        os.system("mkdir -p " + self._dir_tmp)
        os.system("rm -rf " + self._dir_tmp + "*")


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
        if registration_type not in ["Rigid", "Affine"]:
            raise ValueError("Error: Registration type can only be either 'Rigid' or 'reg_f3d'")
        
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


    ## Get registered image
    #  \return registered image as Stack object
    def get_registered_image(self):
        return self._registered_image


    def run_registration(self, id="", verbose=0):
        self._run_registration(id, verbose)


    def _run_registration(self, id, verbose):

        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

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
        cmd += "--f " + self._dir_tmp + fixed_str + " "
        cmd += "--m " + self._dir_tmp + moving_str + " "
        if self._use_fixed_mask:
            cmd += "--fmask " + self._dir_tmp + fixed_mask_str + " "
        if self._use_moving_mask:
            cmd += "--mmask " + self._dir_tmp + moving_mask_str + " "
        cmd += "--tout " + self._dir_tmp + registration_transform_str + " "
        cmd += "--useAffine " + str(int(self._registration_type is "Affine")) + " "
        cmd += "--useMultires " + str(int(self._use_multiresolution_framework)) + " "
        cmd += "--metric " + self._metric + " "
        cmd += "--scalesEst " + self._scales_estimator + " "
        cmd += "--interpolator " + self._interpolator + " "
        cmd += "--ANTSrad " + str(self._ANTSradius) + " "
        cmd += "--translationScale " + str(self._translation_scale) + " "
        cmd += "--verbose 0 "

        ## Compute oriented Gaussian PSF if desired
        if self._interpolator in ["OrientedGaussian"]:
            ## Get oriented Gaussian covariance matrix
            cov_HR_coord = psf.PSF().get_gaussian_PSF_covariance_matrix_HR_volume_coordinates( self._fixed, self._moving ).flatten()
            cmd += "--cov " + "'" + ' '.join(cov_HR_coord.astype("|S12")) + "'"

        print("\t----- print command -----------------------------")
        print cmd
        print("\t-------------------------------------------------")
        os.system(cmd)

        ## Read transformation file
        params_all = np.loadtxt(self._dir_tmp + registration_transform_str + ".txt")

        parameters_fixed = params_all[0:3]
        parameters = params_all[3:]

        if self._registration_type in ["Rigid"]:
            self._transform_sitk = sitk.Euler3DTransform()
        else:
            self._transform_sitk = sitk.AffineTransform(3)

        self._transform_sitk.SetParameters(parameters)
        self._transform_sitk.SetFixedParameters(parameters_fixed)

        moving_warped_sitk = sitk.Resample(self._moving.sitk, self._fixed.sitk, self._transform_sitk, sitk.sitkLinear, 0.0, self._moving.sitk.GetPixelIDValue())
        sitk.WriteImage(moving_warped_sitk, self._dir_tmp + "RegistrationITK_result.nii.gz")


    def _run_registration_affine(self, id, verbose):

        return None



