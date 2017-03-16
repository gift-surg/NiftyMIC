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
import utilities.PythonHelper as ph
import utilities.SimpleITKHelper as sitkh
import base.PSF as psf
import base.Stack as st

class NiftyReg:

    def __init__(self, fixed=None, moving=None, use_fixed_mask=False, use_moving_mask=False, registration_type="Rigid", registration_method="reg_aladin", options="", verbose=True):
        self._fixed = fixed
        self._moving = moving

        self._use_fixed_mask = use_fixed_mask
        self._use_moving_mask = use_moving_mask

        self._affine_transform_sitk = None
        self._control_point_grid_sitk = None
        self._registered_image = None

        ## Temporary output where files are written in order to use NiftyReg
        self._dir_tmp = "/tmp/NiftyReg/"
        ph.create_directory(self._dir_tmp, delete_files=False)

        self._run_registration = {
            "reg_aladin"    :   self._run_reg_aladin,
            "reg_f3d"       :   self._run_reg_f3d
        }
        self._registration_method = registration_method
        self._registration_type = registration_type

        self._options = options + " "
        self._verbose = verbose


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
    def set_registration_method(self, registration_method):
        if registration_method not in self._run_registration.keys():
            raise ValueError("Error: Registration method not possible'")
        self._registration_method = registration_method

    def get_registration_method(self):
        return self._registration_method


    ##
    # Sets the registration type.
    # \date       2017-02-02 16:42:13+0000
    #
    # \param      self               The object
    # \param      registration_type  The registration type
    #
    def set_registration_type(self, registration_type):
        if registration_type not in ["Rigid", "Affine"]:
            raise ValueError("Error: Registration type not possible")
        self._registration_type = registration_type


    ## Get chosen type of registration used
    #  \return registration type as string
    def get_registration_type(self):
        return self._registration_type


    ## Set options used for either reg_aladin or reg_f3d
    #  \param[in] options as string
    #  \example options="-voff"
    #  \example options="-voff -rigOnly -platf 1"
    def set_options(self, options):
        self._options = options


    ## Get options used for either reg_aladin or reg_f3d
    #  \return options chosen as string
    def get_options(self):
        return self._options


    def use_verbose(self, flag):
        self._verbose = flag

    def get_verbose(self):
        return self._verbose


    ## Get affine transform in (Simple)ITK format after having run reg_aladin
    #  \return affine transform as SimpleITK object
    def get_registration_transform_sitk(self):
        return self._affine_transform_sitk


    ## Get registered image
    #  \return registered image as Stack object
    def get_registered_image(self):
        return self._registered_image


    def run_registration(self):

        ## Clean output directory first
        ph.clear_directory(self._dir_tmp)

        self._run_registration[self._registration_method]()


    ## Run reg_aladin, i.e. Block matching algorithm for global affine registration.
    def _run_reg_aladin(self):

        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

        if self._registration_type in ["Rigid"]:
            self._options += "-rigOnly "

        if not self._verbose:
            self._options += "-voff "

        moving_str = "NiftyReg_moving_" + self._moving.get_filename()
        fixed_str = "NiftyReg_fixed_" + self._fixed.get_filename()
        moving_mask_str = "NiftyReg_moving_mask_" + self._moving.get_filename() 
        fixed_mask_str = "NiftyReg_fixed_mask_" + self._fixed.get_filename()
        
        res_affine_image_str = "NiftyReg_regaladin_WarpImage_" + self._fixed.get_filename() + "_" + self._moving.get_filename()
        res_affine_matrix_str = "NiftyReg_regaladin_WarpMatrix_" + self._fixed.get_filename() + "_" + self._moving.get_filename()

        ## Write images to HDD before they can be used for NiftyReg
        if not os.path.isfile(self._dir_tmp + moving_str + ".nii.gz"):
            sitk.WriteImage(self._moving.sitk, self._dir_tmp + moving_str + ".nii.gz")
        if not os.path.isfile(self._dir_tmp + fixed_str + ".nii.gz"):
            sitk.WriteImage(self._fixed.sitk, self._dir_tmp + fixed_str + ".nii.gz")
        if not os.path.isfile(self._dir_tmp + moving_mask_str + ".nii.gz") and self._use_moving_mask:
            sitk.WriteImage(self._moving.sitk_mask, self._dir_tmp + moving_mask_str + ".nii.gz")
        if not os.path.isfile(self._dir_tmp + fixed_mask_str + ".nii.gz") and self._use_fixed_mask:
            sitk.WriteImage(self._fixed.sitk_mask, self._dir_tmp + fixed_mask_str + ".nii.gz")

        ## Run reg_aladin
        self._reg_aladin(fixed_str, moving_str, fixed_mask_str, moving_mask_str, self._options, res_affine_matrix_str, res_affine_image_str)


    ## Run reg_f3d, i.e. Fast Free-Form Deformation algorithm for non-rigid registration. 
    def _run_reg_f3d(self):
        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

        moving_str = "NiftyReg_moving_" + self._moving.get_filename()
        fixed_str = "NiftyReg_fixed_" + self._fixed.get_filename()
        moving_mask_str = "NiftyReg_moving_mask_" + self._moving.get_filename() 
        fixed_mask_str = "NiftyReg_fixed_mask_" + self._fixed.get_filename()
        
        res_f3d_image_str = "NiftyReg_regf3d_WarpImage_" + self._fixed.get_filename() + "_" + self._moving.get_filename()
        res_f3d_cpp_str = "NiftyReg_regf3d_WarpCpp_" + self._fixed.get_filename() + "_" + self._moving.get_filename()
        
        ## Write images to HDD before they can be used for NiftyReg
        if not os.path.isfile(self._dir_tmp + moving_str + ".nii.gz"):
            sitk.WriteImage(self._moving.sitk, self._dir_tmp + moving_str + ".nii.gz")
        if not os.path.isfile(self._dir_tmp + fixed_str + ".nii.gz"):
            sitk.WriteImage(self._fixed.sitk, self._dir_tmp + fixed_str + ".nii.gz")
        if not os.path.isfile(self._dir_tmp + moving_mask_str + ".nii.gz") and self._use_moving_mask:
            sitk.WriteImage(self._moving.sitk_mask, self._dir_tmp + moving_mask_str + ".nii.gz")
        if not os.path.isfile(self._dir_tmp + fixed_mask_str + ".nii.gz") and self._use_fixed_mask:
            sitk.WriteImage(self._fixed.sitk_mask, self._dir_tmp + fixed_mask_str + ".nii.gz")

        ## Optional: Use reg_aladin before if desired
        affine_matrix_str = None
        # affine_matrix_str = fixed_str + "_affine_matrix_NiftyReg"

        ## Run reg_f3d
        self._reg_f3d(fixed_str, moving_str, fixed_mask_str, moving_mask_str, self._options, affine_matrix_str, res_f3d_image_str, res_f3d_cpp_str)


    ## Block matching algorithm for global affine registration.
    #  \param[in] fixed_str name of fixed image stored in self._dir_tmp folder, string
    #  \param[in] moving_str name of moving image stored in self._dir_tmp folder, string
    #  \param[in] fixed_mask_str name of fixed mask image stored in self._dir_tmp folder, string
    #  \param[in] moving_mask_str name of moving mask image stored in self._dir_tmp folder, string
    #  \param[in] options options used for reg_aladin, string
    #  \param[out] res_affine_matrix_str name of resulting affine transform matrix to be stored in self._dir_tmp, string
    #  \param[out] res_affine_image_str name of resulting registered image being stored in self._dir_tmp, string
    def _reg_aladin(self, fixed_str, moving_str, fixed_mask_str, moving_mask_str, options, res_affine_matrix_str, res_affine_image_str):

        cmd =  "reg_aladin " + options + " "
        cmd += "-ref " + self._dir_tmp + fixed_str + ".nii.gz "
        cmd += "-flo " + self._dir_tmp + moving_str + ".nii.gz "
        if self._use_fixed_mask:
            cmd += "-rmask " + self._dir_tmp + fixed_mask_str + ".nii.gz "
        if self._use_moving_mask:
            cmd += "-fmask " + self._dir_tmp + moving_mask_str + ".nii.gz "
        cmd += "-res " + self._dir_tmp + res_affine_image_str + ".nii.gz "
        cmd += "-aff " + self._dir_tmp + res_affine_matrix_str + ".txt "

        # print(cmd)
        if self._registration_type in ["Rigid"]:
            sys.stdout.write("Rigid registration (NiftyReg reg_aladin) ... ")
        else:
            sys.stdout.write("Affine registration (NiftyReg reg_aladin) ... ")
        sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
        os.system(cmd)
        print "done"

        ## Read trafo and invert such that format fits within SimpleITK structure
        matrix = np.loadtxt(self._dir_tmp + res_affine_matrix_str + ".txt")
        A = matrix[0:-1,0:-1]
        t = matrix[0:-1,-1]

        ## Convert to SimpleITK physical coordinate system
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

        cmd =  "reg_f3d " + options + " "
        cmd += "-ref " + self._dir_tmp + fixed_str + ".nii.gz "
        cmd += "-flo " + self._dir_tmp + moving_str + ".nii.gz "
        if self._use_fixed_mask:
            cmd += "-rmask " + self._dir_tmp + fixed_mask_str + ".nii.gz "
        if self._use_moving_mask:
            cmd += "-fmask " + self._dir_tmp + moving_mask_str + ".nii.gz "
        cmd += "-res " + self._dir_tmp + res_f3d_image_str + ".nii.gz "
        cmd += "-cpp " + self._dir_tmp + res_control_point_grid_str + ".nii.gz"

        if affine_matrix_str is not None:
            cmd += " -aff " + self._dir_tmp + affine_matrix_str + ".txt "

        
        # print(cmd)
        sys.stdout.write("Deformable registration (NiftyReg reg_f3d) ... ")
        
        sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
        os.system(cmd)
        print("done")

        ## Get registered image as Stack object
        self._registered_image = st.Stack.from_filename(self._dir_tmp, res_f3d_image_str)
        # self._registered_image.show()

        ## Get Control Point Grid
        self._control_point_grid_sitk = sitk.ReadImage(self._dir_tmp + res_control_point_grid_str + ".nii.gz")
        # sitkh.show_sitk_image(self._control_point_grid_sitk)


