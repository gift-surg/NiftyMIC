## \file DataPreprocessing.py
#  \brief  
# 
#  \author Michael Ebner
#  \date July 2015

## Import libraries
import os                       # used to execute terminal commands in python
import numpy as np
import sys

## Import other py-files within src-folder
from SliceStack import *

class DataPreprocessing:

    def __init__(self, dir_out, dir_in, filenames):

        print("\n***** Data preprocessing (DP) ******")

        self._N_stacks = len(filenames)

        self._stacks = [None]*self._N_stacks
        self._masks = [None]*self._N_stacks
        
        self._dir_results = dir_out


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

            ## Link image to stack
            self._stacks[i] = SliceStack(self._dir_results_input_data, new_name)

            ## 2) Copy masks:

            cmd = "cp " + dir_in + filenames[i] + "_mask.nii.gz " \
                        + self._dir_results_input_data + new_name + "_mask.nii.gz"
            # print cmd
            os.system(cmd)
            
            if os.path.isfile(self._dir_results_input_data + new_name + "_mask.nii.gz"):
                ## Link mask to stack
                self._masks[i] = SliceStack(self._dir_results_input_data, new_name + "_mask")

        print(str(self._N_stacks) + " stacks were copied to directory " + self._dir_results_input_data)


        return None


    def segmentation_propagation(self, target_stack_id):

        print("\n** DP: Segmentation progation **\n")

        ## Create folder if not already existing
        os.system("mkdir -p " + self._dir_results_seg_prop)
        # os.system("mkdir -p " + self._dir_results_input_data_dp_final) 

        ## Copy files to _dir_results_seg_prop:
        # stacks_dp = self._copy_stacks_to_directory(self._dir_results_seg_prop)

        ## Compute propagation of masks based on target stack
        try:
            ## Define floating image details
            dir_flo = self._stacks[target_stack_id].get_dir()
            flo_image = self._stacks[target_stack_id].get_filename()
            flo_mask = self._masks[target_stack_id].get_dir() \
                        + self._masks[target_stack_id].get_filename()


            methods = ["NiftyReg", "FLIRT"]
            method = methods[1]

            ## Output directory of computed results
            dir_res = self._dir_results_seg_prop

            ## Counter for current image processing
            ctr = 0

            for i in range(0,self._N_stacks):

                ## Define floating image details
                dir_ref = self._stacks[i].get_dir()
                ref_image = self._stacks[i].get_filename()

                ## Used intermediate variable names
                res_affine_image = "ref_" + ref_image + "_flo_" \
                            + flo_image + "_affine_result"
                res_affine_matrix = "ref_" + ref_image + "_flo_" \
                            + flo_image + "_affine_matrix"
                res_nrr_image    = "ref_" + ref_image + "_flo_" \
                            + flo_image + "_nrr_result"
                res_nrr_cpp = "ref_" + ref_image + "_flo_" \
                            + flo_image + "_nrr_cpp"

                ## Define name of obtained propagated mask of floating image
                ## in reference space
                res_mask_flo = "ref_" + ref_image + "_flo_" + flo_image + "_mask"

                if i == target_stack_id:
                    continue

                ctr += 1

                ## Affine registration
                if method == "FLIRT":
                    options = "-usesqform "
                    cmd = "flirt " + options + \
                        "-ref " + dir_ref + ref_image + ".nii.gz " + \
                        "-in " + dir_flo + flo_image + ".nii.gz " + \
                        "-out " + dir_res + res_affine_image + ".nii.gz " + \
                        "-omat " + dir_res + res_affine_matrix + ".txt"
                    sys.stdout.write("Rigid registration (FLIRT) " + str(ctr) + "/" + str(self._N_stacks-1) + " ... ")

                    # img = SliceStack(dir_res, res_affine_image)
                    # T = np.loadtxt(dir_res + res_affine_matrix + ".txt")
                    # np.savetxt(dir_res + res_affine_matrix + ".txt", np.linalg.inv(T))

                else:
                    ## NiftyReg: 1) Global affine registration of reference image:
                    #  \param[in] -ref reference image
                    #  \param[in] -flo floating image
                    #  \param[out] -res affine registration of floating image
                    #  \param[out] -aff affine transformation matrix
                    options = "-voff "
                    # options = "-voff -platf Cuda=1 "
                    cmd = "reg_aladin " + options + \
                        "-ref " + dir_ref + ref_image + ".nii.gz " + \
                        "-flo " + dir_flo + flo_image + ".nii.gz " + \
                        "-rmask " + self._masks[i].get_dir() + ref_image + "_mask.nii.gz " + \
                        "-fmask " + flo_mask + ".nii.gz " + \
                        "-res " + dir_res + res_affine_image + ".nii.gz " + \
                        "-aff " + dir_res + res_affine_matrix + ".txt"
                    sys.stdout.write("Rigid registration (NiftyReg reg_aladin) " + str(ctr) + "/" + str(self._N_stacks-1) + " ... ")

                # print(cmd)
                sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
                # print(cmd)
                os.system(cmd)
                print "done"

                # affine = np.loadtxt(dir_results + res_image_affine + ".txt")
                # stacks_dp[i].set_affine(affine)

                ## NiftyReg: 2) Non-rigid registration:
                #  \param[in] options (like 'be' and 'sx')
                #  \param[in] -ref reference image
                #  \param[in] -flo floating image (templates)
                #  \param[in] -aff affine transformation matrix
                #  \param[out] -res non-rigid registration of floating image
                #  \param[out] -cpp control point grid
                options = "-voff "
                cmd = "reg_f3d " + options + \
                    "-ref " + dir_ref + ref_image + ".nii.gz " + \
                    "-flo " + dir_flo + flo_image + ".nii.gz " + \
                    "-aff " + dir_res + res_affine_matrix + ".txt " + \
                    "-res " + dir_res + res_nrr_image + ".nii.gz " + \
                    "-cpp " + dir_res + res_nrr_cpp + ".nii.gz"
                sys.stdout.write("Non-rigid registration (NiftyReg reg_f3d) " + str(ctr) + "/" + str(self._N_stacks-1) + " ... ")
                sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
                # print(cmd)
                # os.system(cmd)
                # print "done"
                


                ## NiftyReg: 3) Propagate labels of floating image into space 
                #               of reference image based on non-rigid 
                #               transformation parametrization stored in
                #               *cpp.nii.gz:
                #  \param[in] -inter interpolation order
                #  \param[in] -ref reference image
                #  \param[in] -flo floating image
                #  \param[in] -trans control point grid
                #  \param[out] -res propagated labels of flo image in ref-space
                options = "-voff "
                    # "-trans " + dir_res + res_nrr_cpp + ".nii.gz " + \
                cmd = "reg_resample " + options + "-inter 0 " + \
                    "-ref " + dir_ref + ref_image + ".nii.gz " + \
                    "-flo " + flo_mask + ".nii.gz " + \
                    "-trans " + dir_res + res_affine_matrix + ".txt " + \
                    "-res " + dir_res + res_mask_flo + ".nii.gz"
                sys.stdout.write("Resampling (NiftyReg reg_resample) " + str(ctr) + "/" + str(self._N_stacks-1) + " ... ")
                sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
                # print(cmd)
                os.system(cmd)
                print("done")

                self._masks[i] = SliceStack(dir_res, res_mask_flo)
                print("Propagated labels were copied to directory " + dir_res)

            for i in range(0, self._N_stacks):
                print(str(self._masks[i].get_dir()) + str(self._masks[i].get_filename()))


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


    def cap_and_normalize_images(self):
        print("\n** DP: Normalization of Images **\n")

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
