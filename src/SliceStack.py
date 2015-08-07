## \file SliceStack.py
#  \brief Class generated to handle a 3D image built of the stack of 2D slices.
# 
#  \author Michael Ebner
#  \date June 2015

## Import libraries
import nibabel as nib           # nifti files
import numpy as np
import copy
import os                       # used to execute terminal commands in python


## Import other py-files within src-folder
from TwoDImage import *



class SliceStack:

    def __init__(self, dir_nifti, nifti_filename):
        self._dir_nifti = dir_nifti
        self._nifti_filename = nifti_filename
        self._img_nifti = nib.load(dir_nifti + nifti_filename + ".nii.gz")

        self._affine = self._img_nifti.affine
        self._data = self._img_nifti.get_data()
        self._header = self._img_nifti.header

        self._N_slices = self._img_nifti.get_data().shape[2]
        self._single_2d_slices = [None]*self._N_slices


    # @classmethod
    # def directly_from_nifti(self, img_nifti):
    #     # Initialize directly from nifti file without loading it from HDD
        
    #     self._img_nifti = img_nifti

    #     self._affine = self._img_nifti.affine
    #     self._dir_nifti = None
    #     self._nifti_filename = None


    # def __copy__(self):
    #     return SliceStack(self._dir_nifti, self._nifti_filename)


    # fails
    # def __deepcopy__(self, memo):
    #     return SliceStack(copy.deepcopy(self._dir_nifti, 
    #         self._nifti_filename, memo))

    def copy_stack(self, name):
        nifti = nib.Nifti1Image(self._data, affine=self._affine, header=self._header)
        nib.save(nifti, self._dir_nifti + self._nifti_filename + "_" + name + ".nii.gz")

        SliceStack(self._dir_nifti, self._nifti_filename + "_" + name)


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


    ## Burst 3D image (stack of slices) into its single slices such that each 
    #  slice is stored in a seperate nifti-file
    def burst_into_single_slices(self):
        dir_out_single_slice = self._dir_nifti + "single_slices/"

        ## Create folders if not already existing
        os.system("mkdir -p " + dir_out_single_slice)

        affine = self._affine
        trafo = np.identity(4)
        # data = np.zeros(self._img_nifti.get_data().shape)

        ## Affine matrix needed to indicate slice in scanner coordinates:
        #  Basically a mere translation towards through-plane axis. Hence,
        #  get direction of through-plane axis by substracting the coordinates
        #  of first pixel of slice 1 from the one of slice 0
        v0 = affine.dot([0,0,0,1])
        v1 = affine.dot([0,0,1,1])
        t = (v1-v0)[0:-1]           # Translation vector to address current slice


        # for i in xrange(0,3):
        for i in xrange(0,self._N_slices):
            trafo[0:3,3] = i*t
            filename = self._nifti_filename + "_" + str(i)
            # data *= 0            
            # data[:,:,i] = self._data[:,:,i]
            data = self._data[:,:,i]

            nifti = nib.Nifti1Image(data, affine=np.dot(trafo, affine), header=self._header)
            nib.save(nifti, dir_out_single_slice +  filename + ".nii.gz") 

            print("Single slice %d/%d of image %s.nii.gz was saved to directory %s" % 
                (i+1, self._N_slices, self._nifti_filename, dir_out_single_slice))

            self._single_2d_slices[i] = TwoDImage(dir_out_single_slice, filename)
            
        return None


    def get_single_slices(self):

        ## Check whether list self._single_2d_slices only contains only Nones,
        #  i.e. stack was not bursted into its slices
        if all(x is None for x in self._single_2d_slices):
            self.burst_into_single_slices()

        return self._single_2d_slices


    def set_affine(self, affine):
        try:
            if (affine.shape != (4, 4)):
                raise ValueError("Affine transformation non-compliant")

            self._affine = affine

            nifti = nib.Nifti1Image(self._data, affine=self._affine, header=self._header)
            nib.save(nifti, self._dir_nifti + self._nifti_filename + ".nii.gz") 
            print("Affine transformation of image " + self._nifti_filename + " successfully updated")

        except ValueError as err:
            print(err.args)


    def set_data(self, array):
        try:
            if (array.size != self._data.size):
                raise ValueError("Dimension mismatch of data array")

            self._data = array

            nifti = nib.Nifti1Image(self._data, affine=None, header=self._header)
            nib.save(nifti, self._dir_nifti + self._nifti_filename + ".nii.gz") 
            print("Data array of image " + self._nifti_filename + " successfully updated")          

        except ValueError as err:
            print(err.args)

