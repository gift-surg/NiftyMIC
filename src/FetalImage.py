## \file fetalImage.py
#  \brief  
# 
#  \author Michael Ebner
#  \date June 2015

import nibabel as nib           # nifti files
import numpy as np


class FetalImage:

    def __init__(self, dir_nifti, nifti_filename):
        self._dir_nifti = dir_nifti
        self._nifti_filename = nifti_filename
        self._input_nifti = nib.load(dir_nifti + nifti_filename + ".nii.gz")

    def get_data(self):
        return self.input_nifti.get_data()

    def get_dir(self):
        return self._dir_nifti

    def get_filename(self):
        return self._nifti_filename

    