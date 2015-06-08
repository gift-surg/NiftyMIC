## \file Registration.py
#  \brief  
# 
#  \author Michael Ebner
#  \date June 2015

## Import libraries
import nibabel as nib           # nifti files
import numpy as np
import os                       # used to execute terminal commands in python

## Import other py-files within src-folder
from FetalImage import *

class Registration:

    def __init__(self, fetal_stacks, target_stack_id):
        self._fetal_stacks = fetal_stacks
        self._target_stack = fetal_stacks[target_stack_id]
        self._registered_stacks = []
        self._registered_stacks.append(self._target_stack)
        self._floating_stacks = []

        for i in range(0,len(fetal_stacks)):
            if (i != target_stack_id):
                self._floating_stacks.append(fetal_stacks[i])


    def get_fetal_stacks(self):
        return self._fetal_stacks

    def get_floating_stacks(self):
        return self._floating_stacks

    def get_target_stack(self):
        return self._target_stack

    def get_registered_stacks(self):
        return self._registered_stacks

    def register_images(self):
        dir_out = "../results/tmp/"
        N_floatings = len(self._floating_stacks)

        ## create folder if not already existing
        os.system("mkdir -p " + dir_out)

        dir_ref = self._target_stack.get_dir()
        ref_filename = self._target_stack.get_filename()

        for i in range(0, N_floatings):
            dir_flo = self._floating_stacks[i].get_dir()
            flo_filename = self._floating_stacks[i].get_filename()
            res_filename = "ref_" + ref_filename + "_flo_" \
                        + flo_filename + "_affine_result"
            affine_matrix_filename = "ref_" + ref_filename + "_flo_" \
                        + flo_filename + "_affine_matrix"


            if not os.path.isfile(dir_out + res_filename + ".nii.gz"):
                ## NiftyReg: 1) Global affine registration of reference image:
                #  \param[in] -ref reference image
                #  \param[in] -flo floating image
                #  \param[out] -res affine registration of floating image
                #  \param[out] -aff affine transformation matrix
                cmd = "reg_aladin " + \
                    "-ref " + dir_ref + ref_filename + ".nii.gz " + \
                    "-flo " + dir_flo + flo_filename + ".nii.gz " + \
                    "-res " + dir_out + res_filename + ".nii.gz " + \
                    "-aff " + dir_out + affine_matrix_filename + ".txt"

                # print(cmd)
                os.system(cmd)

            self._registered_stacks.append(FetalImage(dir_out,res_filename))



