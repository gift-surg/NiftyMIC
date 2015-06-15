## \file fetalImage.py
#  \brief  
# 
#  \author Michael Ebner
#  \date June 2015

import nibabel as nib           # nifti files
import numpy as np
import copy


class FetalImage:

    def __init__(self, dir_nifti, nifti_filename):
        self._dir_nifti = dir_nifti
        self._nifti_filename = nifti_filename
        self._img_nifti = nib.load(dir_nifti + nifti_filename + ".nii.gz")

        self._affine = self._img_nifti.affine
        self._data = self._img_nifti.get_data()
        self._header = self._img_nifti.header

    # @classmethod
    # def directly_from_nifti(self, img_nifti):
    #     # Initialize directly from nifti file without loading it from HDD
        
    #     self._img_nifti = img_nifti

    #     self._affine = self._img_nifti.affine
    #     self._dir_nifti = None
    #     self._nifti_filename = None


    # def __copy__(self):
    #     return FetalImage(self._dir_nifti, self._nifti_filename)


    # fails
    # def __deepcopy__(self, memo):
    #     return FetalImage(copy.deepcopy(self._dir_nifti, 
    #         self._nifti_filename, memo))


    def get_shape(self):
        return self._img_nifti.get_data().shape


    def get_dir(self):
        return self._dir_nifti


    def get_filename(self):
        return self._nifti_filename


    def get_header(self):
        return self._header


    def get_data(self):
        return self._data


    def get_affine(self):
        return self._affine


    def set_data(self, array):
        try:
            if (array.size != self._data.size):
                raise ValueError("Dimension mismatch of data array")

            self._data = array
            self._update_file()

        except ValueError as err:
            print(err.args)


    def _update_file(self):
        nifti = nib.Nifti1Image(self._data, affine=None, header=self._header)
        nib.save(nifti, self._dir_nifti + self._nifti_filename + ".nii.gz") 
        print("File " + self._nifti_filename + " successfully updated.")          
