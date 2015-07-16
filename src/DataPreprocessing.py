## \file DataPreprocessing.py
#  \brief  
# 
#  \author Michael Ebner
#  \date July 2015

## Import libraries
import os                       # used to execute terminal commands in python
import numpy as np

## Import other py-files within src-folder
from SliceStack import *

class DataPreprocessing:

    def __init__(self, dir_out, dir_in, filenames):

        print("\n***** Data preprocessing (DP): ******")

        self._stacks = []
        self._masks = []
        
        self._dir_results = dir_out
        self._N_stacks = len(filenames)


        ## Directory of input data used for reconstruction algorithm
        self._dir_results_input_data = self._dir_results + "input_data/"

        ## Optional: Directory of (intermediate) segmentation propagation data:
        self._dir_results_seg_prop = self._dir_results + "input_data_segmentation_prop/"

        ## Optional: Directory after all DP steps ready for reconstruction algorthm:
        self._dir_results_input_data_dp_final = self._dir_results + "input_data_dp_final/"


        ## Create folder if not already existing
        os.system("mkdir -p " + self._dir_results_input_data)


        ## Copy files to dir_results_input_data and load them into _stacks:
        ## (Filenames represent continouous numbering)
        for i in range(0, self._N_stacks):
            new_name = str(i)

            ## 1) Copy images:
            cmd = "cp " + dir_in + filenames[i] + ".nii.gz " \
                        + self._dir_results_input_data + new_name + ".nii.gz"
            # print cmd
            os.system(cmd)

            ## Append stack
            self._stacks.append(SliceStack(self._dir_results_input_data, new_name))

            ## 2) Copy masks:

            cmd = "cp " + dir_in + filenames[i] + "_mask.nii.gz " \
                        + self._dir_results_input_data + new_name + "_mask.nii.gz"
            # print cmd
            os.system(cmd)
            
            if os.path.isfile(self._dir_results_input_data + new_name + "_mask.nii.gz"):
                ## Append mask
                self._masks.append(SliceStack(self._dir_results_input_data, new_name + "_mask"))

            print("Stacks were copied to directory " + self._dir_results_input_data)


        return None


    def segmentation_propagation(self, target_stack_id):

        print("\nDP: Segmentation progation:\n")

        ## Create folder if not already existing
        os.system("mkdir -p " + self._dir_results_seg_prop)
        # os.system("mkdir -p " + self._dir_results_input_data_dp_final) 

        ## Copy files to _dir_results_seg_prop:
        # stacks_dp = self._copy_stacks_to_directory(self._dir_results_seg_prop)

        ## Compute propagation of masks based on target stack
        try:
            ref_image = self._stacks[target_stack_id].get_filename()
            ctr = 0

            for i in range(0,self._N_stacks):

                flo_image = self._stacks[i].get_filename()
                res_image = "ref_" + ref_image + "_flo_" \
                            + flo_image + "_affine_result"
                res_image_affine = "ref_" + ref_image + "_flo_" \
                            + flo_image + "_affine_matrix"

                if i==target_stack_id:
                    continue

                # options = "-voff -platf Cuda=1 "
                options = "-voff "

                ## NiftyReg: 1) Global affine registration of reference image:
                #  \param[in] -ref reference image
                #  \param[in] -flo floating image
                #  \param[out] -res affine registration of floating image
                #  \param[out] -aff affine transformation matrix
                cmd = "reg_aladin " + options + \
                    "-ref " + self._stacks[i].get_dir() + ref_image + ".nii.gz " + \
                    "-rmask " + self._masks[target_stack_id].get_dir() + ref_image + "_mask.nii.gz " + \
                    "-flo " + self._stacks[i].get_dir() + flo_image + ".nii.gz " + \
                    "-fmask " + self._masks[i].get_dir() + flo_image + "_mask.nii.gz " + \
                    "-res " + self._dir_results_seg_prop + res_image + ".nii.gz " + \
                    "-aff " + self._dir_results_seg_prop + res_image_affine + ".txt"
                print(cmd)
                print "Rigid registration %d/%d ... " % (ctr+1,self._N_stacks-1)
                os.system(cmd)
                print "Rigid registration %d/%d ... done" % (ctr+1,self._N_stacks-1)
                ctr+=1

                affine = np.loadtxt(self._dir_results_seg_prop + res_image_affine + ".txt")
                # stacks_dp[i].set_affine(affine)


        except ValueError as err:
            print(err.args)

        return None

    def crop_and_copy_images(self):

        ## Create folder if not already existing
        os.system("mkdir -p " + self._dir_results_input_data_dp_final)

        ## Crop images


        ## Copy images
        for i in range(0, self._N_stacks):
            filename = self._stacks[i].get_filename()
            cmd = "cp " + self._stacks[i].get_dir() + filename + ".nii.gz " \
                        + self._dir_results_input_data_dp_final+ filename + ".nii.gz"
            # print cmd
            os.system(cmd)

        return None


    def normalize_images(self):
        print("\nDP: Normalization of Images:\n")

        ## Create folder if not already existing
        os.system("mkdir -p " + self._dir_results_input_data_dp_final)

        self._stacks = self._copy_stacks_to_directory(self._dir_results_input_data_dp_final)

        for i in range(0, self._N_stacks):  
            
            data = self._stacks[i].get_data()
            minimal_value = np.min(data)

            ## Set maximum intensity to value of percentile
            cap = np.percentile(data, 99.9)

            ## Intensities greater than cap are clipped to cap
            data = np.clip(data, np.min(data), cap)

            ## Normalize the intensities between 0 and 255 (integer-valued)

            if minimal_value < 0 or cap > 255:
                data = 255*(data - minimal_value) / float(cap - minimal_value)

            self._stacks[i].set_data(data)

        return None


    def create_rectangular_mask(self, dir, filename, center, width, height):

        cmd = "cp " + dir + filename + ".nii.gz " \
                    + self._dir_results_input_data + filename + "_mask.nii.gz"
        os.system(cmd)

        mask = SliceStack(self._dir_results_input_data, filename + "_mask")

        data = np.zeros(mask.get_shape())

        left = np.round(center[0]-width/2)
        right = np.round(center[0]+width/2)
        bottom  = np.round(center[1]-height/2)
        top  = np.round(center[1]+height/2)

        data[left:right, bottom:top, :] = 1

        mask.set_data(data)


    def get_stacks(self):
        return self._stacks


    def _copy_stacks_to_directory(self, dir_out):
        stacks = []

        ## Load directory of current stacks
        dir_in = self._stacks[0].get_dir()

        ## Copy files to dir_out:
        for i in range(0, self._N_stacks):
            filename = self._stacks[i].get_filename()
            cmd = "cp " + dir_in + filename + ".nii.gz " \
                        + dir_out + filename + ".nii.gz"
            # print cmd
            os.system(cmd)

            ## Load stacks
            stacks.append(SliceStack(dir_out, filename))

        return stacks


    ## NiftyReg only allows very short filenames => filenames needed to be renamed
    # def _rename_files(self):
    #     for i in range(0, self._N_stacks):
    #         dir = self._stacks[i].get_dir()
    #         cmd = "mv " + dir + self._stacks[i].get_filename() + ".nii.gz " \
    #                     + dir + str(i) + ".nii.gz"
    #         os.system(cmd)
    #         self._stacks[i] = SliceStack(dir, str(i))
