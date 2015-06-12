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
        self._floating_stacks = []
        self._registered_stacks = []
        self._HR_stack \
            = self.resample_to_isotropic_grid(fetal_stacks[target_stack_id])

        for i in range(0,len(fetal_stacks)):
            # if (i != target_stack_id):
            self._floating_stacks.append(fetal_stacks[i])


    def get_fetal_stacks(self):
        return self._fetal_stacks

    def get_floating_stacks(self):
        return self._floating_stacks

    def get_target_stack(self):
        return self._target_stack

    def get_HR_stack(self):
        return self._HR_stack

    def get_registered_stacks(self):
        return self._registered_stacks

    def register_images(self):
        dir_out = "../results/tmp/"
        N_floatings = len(self._floating_stacks)

        print N_floatings

        ## create folder if not already existing
        os.system("mkdir -p " + dir_out)

        dir_ref = self._HR_stack.get_dir()
        ref_filename = self._HR_stack.get_filename()

        for i in range(0, N_floatings):
            dir_flo = self._floating_stacks[i].get_dir()
            flo_filename = self._floating_stacks[i].get_filename()
            res_filename = "ref_" + ref_filename + "_flo_" \
                        + flo_filename + "_affine_result"
            affine_matrix_filename = "ref_" + ref_filename + "_flo_" \
                        + flo_filename + "_affine_matrix"


            if not os.path.isfile(dir_out + res_filename + ".nii.gz"):
                ## NiftyReg: Global affine registration of reference image:
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


    def resample_to_isotropic_grid(self, img):
        dir_out = "../results/tmp/"

        header = img.get_header()
        shape = header.get_data_shape()
        pixdim =  header.get_zooms()      # voxel sizes in mm
        data = img.get_data()
        # sform = header.get_sform()
        # qform = header.get_qform()
        # print sform

        length_through_plane = shape[2]*pixdim[2]
        number_of_slices_target_through_plane \
                = np.ceil(length_through_plane / float(pixdim[0])).astype(np.uint8)


        # print("shape = " + str(shape))
        # print("pixdim = " + str(pixdim) + " mm")
        # print("length_through_plane = " + str(length_through_plane) + " mm")
        # print img.get_data().shape
        # print img.affine

        # print(header.get_data_shape())

        # sform[2,0:2] = sform[2,0:2] * length_through_plane/pixdim[0]

        # print sform

        data_target = np.zeros((shape[0],shape[1], number_of_slices_target_through_plane),
            dtype=np.int16)

        i = 0
        for j in range(0,number_of_slices_target_through_plane):
            if (j > (i+1)*pixdim[2]/pixdim[0] and i < shape[2]-1):
                i = i+1
                # print("j="+str(j))
                # print("i="+str(i))
                # print("dist_fine = " + str((j+0.5)*pixdim[0]) )
                # print("dist_coarse = " + str((i)*pixdim[2]) )
                # print("\n")
            data_target[:,:,j] = data[:,:,i]

        img_target_nib = nib.Nifti1Image(data_target, img.affine)
        header_target = img_target_nib.get_header()
        header_target.set_zooms(pixdim[0]*np.ones(3))
        header_target.set_data_shape([shape[0],shape[1],number_of_slices_target_through_plane])
        # header_target.set_xyzt_units(2)
        # header_target.set_sform(sform,code=1)
        # header_target.set_qform(qform)

        # print(header_target.get_zooms())

        img_target_nifti_filename = img.get_filename()
        nib.save(img_target_nib, dir_out + img_target_nifti_filename + ".nii.gz")
        img_target = FetalImage(dir_out, img_target_nifti_filename)

        ## NiftyReg: Resample to isotropic grid
        #  \param[in] -inter interpolation order
        #  \param[in] -ref reference image
        #  \param[in] -flo floating image
        #  \param[in] -trans segmentation of floating image
        #  \param[out] -res propagated labels of flo image in ref-space
        # cmd = "reg_resample " + "-inter 0 " + \
        #     "-ref " + img_target_dummy.get_dir() + img_target_dummy.get_filename() + ".nii.gz " + \
        #     "-flo " + img.get_dir() + img.get_filename() + ".nii.gz " + \
        #     "-trans " + dir_out + "test-trafo.txt " \
        #     "-res " + dir_out + img_target_nifti_filename + ".nii.gz"
        # os.system(cmd)



        # nib.save(img_target, 'my_image.nii.gz')
        # print(img.zooms)

        return img_target

