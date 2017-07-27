# \file FLIRT.py
# \brief      This class makes FLIRT accessible via Python
#
# This class requires Convert3D Medical Image Processing Tool to be installed
# (https://sourceforge.net/projects/c3d/files/c3d/Nightly/)
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       May 2016

# Import libraries
import os
import sys
import SimpleITK as sitk
import numpy as np
import scipy

# Import modules from src-folder
import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh
import base.Stack as st

from definitions import DIR_TMP
from definitions import FLIRT_EXE
from definitions import C3D_AFFINE_TOOL_EXE


class FLIRT:

    def __init__(self,
                 fixed=None,
                 moving=None,
                 use_fixed_mask=False,
                 use_moving_mask=False,
                 registration_type="Rigid",
                 options="",
                 use_verbose=True):
        self._fixed = fixed
        self._moving = moving

        self._use_fixed_mask = use_fixed_mask
        self._use_moving_mask = use_moving_mask

        self._affine_transform_sitk = None
        self._control_point_grid_sitk = None
        self._registered_image = None

        # Temporary output where files are written in order to use NiftyReg
        self._dir_tmp = ph.create_directory(os.path.join(DIR_TMP, "FLIRT"),
                                            delete_files=False)

        self._registration_type = registration_type

        self._options = options
        self._verbose = use_verbose

    # Set fixed/reference/target image
    #  \param[in] fixed fixed/reference/target image as Stack object
    def set_fixed(self, fixed):
        self._fixed = fixed

    # Set moving/floating/source image
    #  \param[in] moving moving/floating/source image as Stack object
    def set_moving(self, moving):
        self._moving = moving

    # Specify whether mask shall be used for fixed image
    #  \param[in] flag boolean
    def use_fixed_mask(self, flag):
        self._use_fixed_mask = flag

    # Specify whether mask shall be used for moving image
    #  \param[in] flag boolean
    def use_moving_mask(self, flag):
        self._use_moving_mask = flag

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

    # Get chosen type of registration used
    #  \return registration type as string
    def get_registration_type(self):
        return self._registration_type

    # Set options used for either reg_aladin or reg_f3d
    #  \param[in] options as string
    #  \example options="-voff"
    #  \example options="-voff -rigOnly -platf 1"
    def set_options(self, options):
        self._options = options

    # Get options used for either reg_aladin or reg_f3d
    #  \return options chosen as string
    def get_options(self):
        return self._options

    def use_verbose(self, flag):
        self._verbose = flag

    def get_verbose(self):
        return self._verbose

    # Get affine transform in (Simple)ITK format after having run reg_aladin
    #  \return affine transform as SimpleITK object
    def get_registration_transform_sitk(self):
        return self._affine_transform_sitk

    # Get registered image
    #  \return registered image as Stack object
    def get_corrected_stack(self):
        corrected_stack = st.Stack.from_stack(self._fixed)
        corrected_stack.update_motion_correction(self._affine_transform_sitk)
        return corrected_stack

    # Get registered image
    #  \return registered image as Stack object
    def get_registered_image(self):
        return self._registered_image

    def run_registration(self):

        # Clean output directory first
        ph.clear_directory(self._dir_tmp)

        self._run_registration()

    # Run FLIRT
    def _run_registration(self):

        options = self._options

        if self._fixed is None or self._moving is None:
            raise ValueError("Error: Fixed and moving image not specified")

        if self._registration_type == "Rigid":
            options += "-dof 6 \\\n"

        elif self._registration_type == "Affine":
            options += "-dof 12 \\\n"

        if self._verbose:
            options += "-verbose 1 \\\n"

        moving_str = "FLIRT_moving_" + self._moving.get_filename()
        fixed_str = "FLIRT_fixed_" + self._fixed.get_filename()
        moving_mask_str = "FLIRT_moving_mask_" + self._moving.get_filename()
        fixed_mask_str = "FLIRT_fixed_mask_" + self._fixed.get_filename()

        res_affine_image_str = "FLIRT_WarpImage_" + \
            self._fixed.get_filename() + "_" + self._moving.get_filename()
        res_affine_matrix_str = "FLIRT_WarpMatrix_" + \
            self._fixed.get_filename() + "_" + self._moving.get_filename()
        res_affine_matrix_itk_str = "ITK_WarpMatrix_" + \
            self._fixed.get_filename() + "_" + self._moving.get_filename()

        # Write images to HDD before they can be used for FLIRT
        if not os.path.isfile(self._dir_tmp + moving_str + ".nii.gz"):
            sitk.WriteImage(self._moving.sitk, self._dir_tmp +
                            moving_str + ".nii.gz")
        if not os.path.isfile(self._dir_tmp + fixed_str + ".nii.gz"):
            sitk.WriteImage(self._fixed.sitk, self._dir_tmp +
                            fixed_str + ".nii.gz")
        if not os.path.isfile(self._dir_tmp + moving_mask_str + ".nii.gz") and self._use_moving_mask:
            sitk.WriteImage(self._moving.sitk_mask,
                            self._dir_tmp + moving_mask_str + ".nii.gz")
        if not os.path.isfile(self._dir_tmp + fixed_mask_str + ".nii.gz") and self._use_fixed_mask:
            sitk.WriteImage(self._fixed.sitk_mask,
                            self._dir_tmp + fixed_mask_str + ".nii.gz")

        cmd = FLIRT_EXE + " \\\n"
        cmd += "-in " + self._dir_tmp + moving_str + ".nii.gz \\\n"
        cmd += "-ref " + self._dir_tmp + fixed_str + ".nii.gz \\\n"
        if self._use_fixed_mask:
            cmd += "-refweight " + self._dir_tmp + fixed_mask_str + ".nii.gz \\\n"
        if self._use_moving_mask:
            cmd += "-inweight " + self._dir_tmp + moving_mask_str + ".nii.gz \\\n"
        cmd += "-out " + self._dir_tmp + res_affine_image_str + ".nii.gz \\\n"
        cmd += "-omat " + self._dir_tmp + res_affine_matrix_str + ".txt \\\n"
        cmd += options

        # print(cmd)
        if self._registration_type == "Rigid":
            sys.stdout.write("Rigid registration (FLIRT) ... ")
        elif self._registration_type == "Affine":
            sys.stdout.write("Affine registration (FLIRT) ... ")
        # flush output; otherwise sys.stdout.write would wait until next
        # newline before printing
        sys.stdout.flush()
        os.system(cmd)
        print("done")

        # Convert FSL to ITK transform
        # Source: https://sourceforge.net/p/advants/discussion/840261/thread/5f5e054f/
        cmd = C3D_AFFINE_TOOL_EXE + " "
        cmd += "-ref " + self._dir_tmp + fixed_str + ".nii.gz "
        cmd += "-src " + self._dir_tmp + moving_str + ".nii.gz "
        cmd += self._dir_tmp + res_affine_matrix_str + ".txt "
        cmd += "-fsl2ras "
        cmd += "-oitk " + self._dir_tmp + res_affine_matrix_itk_str + ".txt"
        os.system(cmd)

        trafo_sitk = sitk.ReadTransform(self._dir_tmp + res_affine_matrix_itk_str + ".txt")
        self._affine_transform_sitk = sitk.AffineTransform(3)
        self._affine_transform_sitk.SetParameters(trafo_sitk.GetParameters())

        # Get registered image as Stack object
        self._registered_image = st.Stack.from_filename(
            os.path.join(self._dir_tmp, res_affine_image_str + ".nii.gz")
        )
