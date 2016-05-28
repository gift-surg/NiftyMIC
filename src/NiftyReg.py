## \file NiftyReg.py
#  \brief This class makes NiftyReg accessible via Python
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016

## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
import SimpleITKHelper as sitkh
import PSF as psf
import Stack as st

class NiftyReg:

    def __init__(self):
        self._moving = None
        self._fixed = None

        self._affine_transform_sitk = None
        self._control_point_grid_sitk = None
        self._registered_image = None

        ## Temporary output where files are written in order to use NiftyReg
        self._dir_tmp = "/tmp/"


    ## Set fixed/reference/target image
    #  \param[in] fixed fixed/reference/target image as Stack object
    def set_fixed(self, fixed):
        self._fixed = fixed


    ## Set moving/floating/source image
    #  \param[in] moving moving/floating/source image as Stack object
    def set_moving(self, moving):
        self._moving = moving


    ## Get affine transform in (Simple)ITK format after having run reg_aladin
    #  \return affine transform as SimpleITK object
    def get_affine_transform_sitk(self):
        return self._affine_transform_sitk


    ## Get registered image
    #  \return registered image as Stack object
    def get_registered_image(self):
        return self._registered_image


    ## Run reg_aladin, i.e. Block matching algorithm for global affine registration.
    #  \param[in] options reg_aladin options as string
    #  \example options="-voff"
    #  \example options="-voff -rigOnly -platf 1"
    def run_reg_aladin(self, options=""):

        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

        moving_str = self._moving.get_filename() + "_moving"
        fixed_str = self._fixed.get_filename() + "_fixed"
        moving_mask_str = self._moving.get_filename() + "_moving_mask"
        fixed_mask_str = self._fixed.get_filename() + "_fixed_mask"
        
        res_affine_matrix_str = fixed_str + "_affine_matrix_NiftyReg"
        res_affine_image_str = moving_str + "_warped_affine_NiftyReg"
        
        ## Write images to HDD before they can be used for NiftyReg
        sitk.WriteImage(self._moving.sitk, self._dir_tmp + moving_str + ".nii.gz")
        sitk.WriteImage(self._fixed.sitk, self._dir_tmp + fixed_str + ".nii.gz")
        sitk.WriteImage(self._moving.sitk_mask, self._dir_tmp + moving_mask_str + ".nii.gz")
        sitk.WriteImage(self._fixed.sitk_mask, self._dir_tmp + fixed_mask_str + ".nii.gz")

        ## Run reg_aladin
        self._reg_aladin(fixed_str, moving_str, fixed_mask_str, moving_mask_str, options, res_affine_matrix_str, res_affine_image_str)


    ## Run reg_f3d, i.e. Fast Free-Form Deformation algorithm for non-rigid registration. 
    #  \param[in] options reg_f3d options as string
    #  \example options="-voff"
    def run_reg_f3d(self, options=""):
        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

        moving_str = self._moving.get_filename() + "_moving"
        fixed_str = self._fixed.get_filename() + "_fixed"
        moving_mask_str = self._moving.get_filename() + "_moving_mask"
        fixed_mask_str = self._fixed.get_filename() + "_fixed_mask"
        
        res_f3d_image_str = moving_str + "_warped_f3d_NiftyReg"
        res_f3d_cpp_str = moving_str + "_cpp_f3d_NiftyReg"
        
        ## Write images to HDD before they can be used for NiftyReg
        sitk.WriteImage(self._moving.sitk, self._dir_tmp + moving_str + ".nii.gz")
        sitk.WriteImage(self._fixed.sitk, self._dir_tmp + fixed_str + ".nii.gz")
        sitk.WriteImage(self._moving.sitk_mask, self._dir_tmp + moving_mask_str + ".nii.gz")
        sitk.WriteImage(self._fixed.sitk_mask, self._dir_tmp + fixed_mask_str + ".nii.gz")

        ## Optional: Use reg_aladin before if desired
        affine_matrix_str = None
        # affine_matrix_str = fixed_str + "_affine_matrix_NiftyReg"

        ## Run reg_f3d
        self._reg_f3d(fixed_str, moving_str, fixed_mask_str, moving_mask_str, options, affine_matrix_str, res_f3d_image_str, res_f3d_cpp_str)


    ## Block matching algorithm for global affine registration.
    #  \param[in] fixed_str name of fixed image stored in self._dir_tmp folder, string
    #  \param[in] moving_str name of moving image stored in self._dir_tmp folder, string
    #  \param[in] fixed_mask_str name of fixed mask image stored in self._dir_tmp folder, string
    #  \param[in] moving_mask_str name of moving mask image stored in self._dir_tmp folder, string
    #  \param[in] options options used for reg_aladin, string
    #  \param[out] res_affine_matrix_str name of resulting affine transform matrix to be stored in self._dir_tmp, string
    #  \param[out] res_affine_image_str name of resulting registered image being stored in self._dir_tmp, string
    def _reg_aladin(self, fixed_str, moving_str, fixed_mask_str, moving_mask_str, options, res_affine_matrix_str, res_affine_image_str):

        cmd = "reg_aladin " + options + " " + \
            "-ref " + self._dir_tmp + fixed_str + ".nii.gz " + \
            "-flo " + self._dir_tmp + moving_str + ".nii.gz " + \
            "-rmask " + self._dir_tmp + fixed_mask_str + ".nii.gz " + \
            "-fmask " + self._dir_tmp + moving_mask_str + ".nii.gz " + \
            "-res " + self._dir_tmp + res_affine_image_str + ".nii.gz " + \
            "-aff " + self._dir_tmp + res_affine_matrix_str + ".txt "

        # print(cmd)
        sys.stdout.write("  Rigid registration (NiftyReg reg_aladin) ... ")

        sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
        os.system(cmd)
        print "done"

        ## Read trafo and invert such that format fits within SimpleITK structure
        matrix = np.loadtxt(self._dir_tmp + res_affine_matrix_str + ".txt")
        A = matrix[0:-1,0:-1]
        t = matrix[0:-1,-1]

        ## Convert to SimpleITK physical coordinate system
        ## TODO: Unit tests according to SimpleITK_NiftyReg_FLIRT.py
        R = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])

        A = R.dot(A).dot(R)
        t = R.dot(t)

        self._affine_transform_sitk = sitk.AffineTransform(A.flatten(), t)

        ## Get registered image as Stack object
        self._registered_image = st.Stack.from_filename(self._dir_tmp, res_affine_image_str)
        # self._registered_image.show()


    ## Fast Free-Form Deformation algorithm for non-rigid registration. 
    #  \param[in] fixed_str name of fixed image stored in self._dir_tmp folder, string
    #  \param[in] moving_str name of moving image stored in self._dir_tmp folder, string
    #  \param[in] fixed_mask_str name of fixed mask image stored in self._dir_tmp folder, string
    #  \param[in] moving_mask_str name of moving mask image stored in self._dir_tmp folder, string
    #  \param[in] options options used for reg_aladin, string
    #  \param[in] affine_matrix_str name of resulting affine transform matrix being stored in self._dir_tmp, string
    #  \param[out] res_f3d_image_str name of resulting registered image to be stored in self._dir_tmp, string
    #  \param[out] res_control_point_grid_str name of control point grid to be stored in self._dir_tmp, string
    def _reg_f3d(self, fixed_str, moving_str, fixed_mask_str, moving_mask_str, options, affine_matrix_str, res_f3d_image_str, res_control_point_grid_str):
        
        if affine_matrix_str is not None:
            res_affine_image_str = moving_str + "_warped_affine_NiftyReg"
            self._reg_aladin(fixed_str, moving_str, fixed_mask_str, moving_mask_str, options, affine_matrix_str, res_affine_image_str)

        cmd = "reg_f3d " + options + " " + \
            "-ref " + self._dir_tmp + fixed_str + ".nii.gz " + \
            "-flo " + self._dir_tmp + moving_str + ".nii.gz " + \
            "-rmask " + self._dir_tmp + fixed_mask_str + ".nii.gz " + \
            "-fmask " + self._dir_tmp + moving_mask_str + ".nii.gz " + \
            "-res " + self._dir_tmp + res_f3d_image_str + ".nii.gz " + \
            "-cpp " + self._dir_tmp + res_control_point_grid_str + ".nii.gz"

        if affine_matrix_str is not None:
            cmd += " -aff " + self._dir_tmp + affine_matrix_str + ".txt "

        
        # print(cmd)
        sys.stdout.write("Non-rigid registration (NiftyReg reg_f3d) ... ")
        
        sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
        os.system(cmd)
        print("done")

        ## Get registered image as Stack object
        self._registered_image = st.Stack.from_filename(self._dir_tmp, res_f3d_image_str)
        # self._registered_image.show()

        ## Get Control Point Grid
        self._control_point_grid_sitk = sitk.ReadImage(self._dir_tmp + res_control_point_grid_str + ".nii.gz")
        sitkh.show_sitk_image(self._control_point_grid_sitk)


