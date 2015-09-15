## \file Registration.py
#  \brief  
# 
#  \author Michael Ebner
#  \date June 2015


## Import libraries
import nibabel as nib           # nifti files
import numpy as np
import numpy.linalg as npl
import os                       # used to execute terminal commands in python
import sys


## Import other py-files within src-folder
from SliceStack import *
from FileAndImageHelpers import *


## Global variables to store computed results
global dir_out
global dir_out_rigid_corr

dir_out = "../results/"
dir_out_rigid_corr = dir_out+"input_data_rigidly_corrected/"

## \brief some description
class Registration:

    def __init__(self, stacks, masks, target_stack_id):

        print("\n***** Rigid Registration (RR) ******")

        ## Create folders if not already existing
        os.system("mkdir -p " + dir_out)
        os.system("mkdir -p " + dir_out_rigid_corr)


        ## Initialize variables
        self._input_stacks = stacks
        self._target_stack = stacks[target_stack_id]
        self._corrected_stacks = [None]*len(self._input_stacks)
        self._masks = masks
        self._N_stacks = len(self._input_stacks)


        try:
            """
            Initialize corrected stacks:
            """
            for i in range(0,len(stacks)):
                filename = self._input_stacks[i].get_filename()

                ## Try to read already corrected stacks
                if  os.path.isfile(dir_out_rigid_corr + filename + ".nii.gz"):
                    # raise ValueError("Corrected stacks already exist!")
                    print("Corrected stack %d was read from directory %s" % (i,dir_out_rigid_corr))

                ## Or copy originally acquired stacks
                else:

                    cmd = "cp " + self._input_stacks[i].get_dir() \
                                + self._input_stacks[i].get_filename() + ".nii.gz " \
                                + dir_out_rigid_corr + filename + ".nii.gz"
                    # print cmd
                    os.system(cmd)

                self._corrected_stacks[i] = SliceStack(dir_out_rigid_corr, filename)


            """
            Initialize estimate of HR volume
            """
            ## HR volume depends on target stack => assign ID to HR volume filename
            filename = "HR_volume_" + self._target_stack.get_filename()

            ## Try to read already existing HR volume estimate
            if os.path.isfile(dir_out + filename + ".nii.gz"):
                # raise ValueError("HR volume already exists!")
                self._HR_volume = SliceStack(dir_out, filename)
                print ("HR volume was read from directory " + dir_out)

            ## Or resample target stack to isotropic grid as first estimate of HRV
            else:
                self._HR_volume = self.resample_to_isotropic_grid(stacks[target_stack_id], filename)
                print ("HR volume initialized as resampled target stack")


        except ValueError as err:
            print(err.args)



    def get_stacks(self):
        return self._input_stacks


    def get_target_stack(self):
        return self._target_stack


    def get_HR_volume(self):
        return self._HR_volume


    def get_corrected_stacks(self):
        return self._corrected_stacks


    ## Register stacks rigidly to current HR volume
    def register_images(self):

        ## Fetch data of current HR volume
        dir_ref = self._HR_volume.get_dir()
        ref_image = self._HR_volume.get_filename()

        methods = ["NiftyReg", "FLIRT"]
        method = methods[1]

        ## Compute rigid registration of each stack by using NiftyReg
        for i in range(0, self._N_stacks):

            ## Fetch data of motion corrupted slices
            dir_flo = self._corrected_stacks[i].get_dir()
            flo_image = self._corrected_stacks[i].get_filename()

            ## Define filenames of resulting rigidly registered stacks
            dir_res = dir_flo
            res_affine_image = flo_image
            res_affine_matrix = flo_image + "_ref_" + ref_image + "_flo_" \
                        + flo_image + "_affine_matrix"


            ## Affine registration
            if method == "FLIRT":
                options = ""
                cmd = "flirt " + options + \
                    "-ref " + dir_ref + ref_image + ".nii.gz " + \
                    "-in " + dir_flo + flo_image + ".nii.gz " + \
                    "-out " + dir_res + res_affine_image + ".nii.gz " + \
                    "-omat " + dir_res + res_affine_matrix + ".txt"
                sys.stdout.write("Rigid registration (FLIRT) to target stack " + str(i+1) + "/" + str(self._N_stacks) + " ... ")

                # img = SliceStack(dir_res, res_affine_image)
                # T = np.loadtxt(dir_res + res_affine_matrix + ".txt")
                # np.savetxt(dir_res + res_affine_matrix + ".txt", np.linalg.inv(T))

            else:
                ## NiftyReg: Global affine registration of reference image:
                #  \param[in] -ref reference image
                #  \param[in] -flo floating image
                #  \param[out] -res affine registration of floating image
                #  \param[out] -aff affine transformation matrix
                options = "-voff "
                # options = "-voff -platf Cuda=1 "
                cmd = "reg_aladin " + options + \
                    "-ref " + dir_ref + ref_image + ".nii.gz " + \
                    "-flo " + dir_flo + flo_image + ".nii.gz " + \
                    "-res " + dir_res + res_affine_image + ".nii.gz " + \
                    "-aff " + dir_res + res_affine_matrix + ".txt"
                sys.stdout.write("Rigid registration (NiftyReg reg_aladin) " + str(i+1) + "/" + str(self._N_stacks) + " ... ")

            # print(cmd)
            sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
            # print(cmd)
            os.system(cmd)
            print "done"

            ## Update results
            self._corrected_stacks[i] = SliceStack(dir_flo, res_affine_image)

        return None


    #\brief Resample image to isotropic grid
    #  \param[in] img instance of SliceStack 
    #  \param[in] filename resampled image is saved as filename.nii.gz in dir_out
    #  \param[in] order interpolation order
    #
    #\details 
    def resample_to_isotropic_grid(self, img, filename, order=0):

        ## Fetch nifti-data for resampling
        header = img.get_header()
        shape = header.get_data_shape()
        pixdim =  header.get_zooms()      # voxel sizes in mm
        data = img.get_data()
        # sform = header.get_sform()
        # qform = header.get_qform()
        # print sform

        ## Compute length of sampling in z-direction and new number of slices
        ## after rescaling their thickness to in-plane resolution
        length_through_plane = shape[2]*pixdim[2]

        ## New number of slices given the new pixel dimension
        number_of_slices_target_through_plane \
                = np.ceil(length_through_plane / float(pixdim[0])).astype(np.uint8)

        ## Incorporation of scaling in affine matrix
        affine = np.identity(4)
        affine[2,2] = pixdim[0]/pixdim[2]
        affine = img.get_affine().dot(affine)

        data_target = np.zeros(
            (shape[0], shape[1], number_of_slices_target_through_plane),
            dtype=np.int16
            )

        """
        ## Nearest neighbour interpolation to uniform grid
        if (order == 0):
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
        """

        ## Store data into nifti-object with appropriate header information
        ## and updated affine matrix
        img_target_nib = nib.Nifti1Image(data_target, affine=affine, header=header)
        
        ## Save nifti-image to directory
        nib.save(img_target_nib, dir_out + filename + ".nii.gz")

        # img_target = 

        ## NiftyReg: Resample to isotropic grid
        #  \param[in] -inter interpolation order
        #  \param[in] -ref reference image
        #  \param[in] -flo floating image
        #  \param[in] -trans segmentation of floating image
        #  \param[out] -res propagated labels of flo image in ref-space
        #
        # reg_resample needs reference to be the right dimension and orientation
        # Basically, only the data is resampled according to the interpolation
        # order
        cmd = "reg_resample -voff " + "-inter " + str(order) + " " \
            "-ref " + dir_out + filename + ".nii.gz " + \
            "-flo " + img.get_dir() + img.get_filename() + ".nii.gz " + \
            "-res " + dir_out + filename + ".nii.gz"
        os.system(cmd)
        print("Resampling of chosen target stack to uniform grid done")

        return SliceStack(dir_out, filename)


    def planar_registering_of_slices(self, reference, target):

        warped = reference

        return warped


    def compute_HR_volume(self):

        data = np.zeros(self._HR_volume.get_data().shape)
        ind = np.zeros(self._HR_volume.get_data().shape)

        for i in range(0, self._N_stacks):
            ## Sum intensities of stacks
            tmp_data = self._corrected_stacks[i].get_data()
            tmp_data = normalize_image(tmp_data)
            data += tmp_data

            ## Store indices of voxels with non-zero contribution
            ind_nonzero = np.nonzero(tmp_data)
            ind[ind_nonzero] += 1

        ## Average over the amount of non-zero contributions of the stack at each index
        ind[ind==0] = 1                 # exclude division by zero
        data = np.divide(data,ind)

        ## Update HR volume
        self._HR_volume.set_data(data)

        return None

