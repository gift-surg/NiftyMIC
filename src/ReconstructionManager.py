## \file ReconstructionManager.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python

## Import modules from src-folder
import StackManager as sm
import InPlaneRigidRegistration as iprr
import FirstEstimateOfHRVolume as efhrv

class ReconstructionManager:

    def __init__(self, dir_output):

        self._dir_results = dir_output
        
        ## Directory of input data used for reconstruction algorithm
        self._dir_results_input_data = self._dir_results + "input_data/"

        ## Directory to store slices and ther computed rigid registration transformations
        self._dir_results_slices = self._dir_results + "slices/"

        ## Optional: Directory of (intermediate) segmentation propagation data:
        # self._dir_results_seg_prop = self._dir_results + "input_data_segmentation_prop/"

        ## Optional: Directory after all DP steps ready for reconstruction algorthm:
        # self._dir_results_input_data_dp_final = self._dir_results + "input_data_dp_final/"


        ## Create folder if not already existing
        os.system("mkdir -p " + self._dir_results)
        os.system("mkdir -p " + self._dir_results_input_data)
        os.system("mkdir -p " + self._dir_results_slices)
        # os.system("mkdir -p " + self._dir_results_seg_prop)
        # os.system("mkdir -p " + self._dir_results_input_data)

        self._filename_reconstructed_volume = "reconstruction"

        self._in_plane_rigid_registration = None
        self._stack_manager = None
        self._HR_volume = None

        return None


    def read_input_data(self, dir_input_to_copy, filenames_to_copy):
        N_stacks = len(filenames_to_copy)
        filenames = [str(i) for i in range(0, N_stacks)]

        self._stack_manager = sm.StackManager()

        ## Copy files to dir_results_input_data:
        ## (Filenames represent continuous numbering)
        for i in range(0, N_stacks):
            new_name = str(filenames[i])

            ## 1) Copy images:
            cmd = "cp " + dir_input_to_copy + filenames_to_copy[i] + ".nii.gz " \
                        + self._dir_results_input_data + filenames[i] + ".nii.gz"
            # print cmd
            os.system(cmd)

            ## 2) Copy masks:
            if os.path.isfile(dir_input_to_copy + filenames_to_copy[i] + "_mask.nii.gz"):
                cmd = "cp " + dir_input_to_copy + filenames_to_copy[i] + "_mask.nii.gz " \
                            + self._dir_results_input_data + filenames[i] + "_mask.nii.gz"
                os.system(cmd)
            
        print(str(N_stacks) + " stacks were copied to directory " + self._dir_results_input_data)

        ## Read stacks:
        self._stack_manager.read_input_data(self._dir_results_input_data, filenames)

        return None


    def compute_first_estimate_of_HR_volume(self):
        first_estimate_of_HR_volume = efhrv.FirstEstimateOfHRVolume(self._stack_manager, self._filename_reconstructed_volume)
        first_estimate_of_HR_volume.compute_first_estimate_of_HR_volume()
        self._HR_volume = first_estimate_of_HR_volume.get_first_estimate_of_HR_volume()
        return None


    def run_in_plane_rigid_registration(self):
        self._in_plane_rigid_registration = iprr.InPlaneRigidRegistration(self._stack_manager)
        return None


    def write_resampled_stacks_after_2D_in_plane_registration(self):
        self._in_plane_rigid_registration.write_resampled_stacks(self._dir_results)
        print("Resampled stacks after in-plane registration successfully written to directory %r " % self._dir_results)
        return None


    def get_stacks(self):
        return self._stack_manager.get_stacks()


    def write_results(self):
        self._stack_manager.write_results(self._dir_results_slices)

        print("All results successfully written to directory %s " % self._dir_results_slices)
        return None