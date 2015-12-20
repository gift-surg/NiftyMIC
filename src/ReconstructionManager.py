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
import SliceToVolumeRegistration as s2vr
import VolumeReconstruction as vr

class ReconstructionManager:

    def __init__(self, dir_output):

        self._dir_results = dir_output
        self._HR_volume_filename = "Reconstruction"
        
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


    def compute_first_estimate_of_HR_volume(self, use_in_plane_registration=False):
        print("\n--- Compute first estimate of HR volume ---")
        
        ## Instantiate object
        first_estimate_of_HR_volume = efhrv.FirstEstimateOfHRVolume(self._stack_manager, self._HR_volume_filename)

        ## Perform estimation of initial HR volume
        first_estimate_of_HR_volume.compute_first_estimate_of_HR_volume(use_in_plane_registration)

        ## Get estimation
        self._HR_volume = first_estimate_of_HR_volume.get_first_estimate_of_HR_volume()
        
        return None


    ## Execute two-step rconstruction alignment approach
    #  \param iterations amount of two-step reconstruction alignment steps
    def run_two_step_reconstruction_alignment_approach(self, iterations=1):
        print("\n--- Run two-step reconstruction alignment approach ---")

        ## Instantiate objects
        slice_to_volume_registration = s2vr.SliceToVolumeRegistration(self._stack_manager)
        volume_reconstruction = vr.VolumeReconstruction(self._stack_manager)


        self._HR_volume.write(directory=self._dir_results, filename=self._HR_volume_filename+"_0")

        ## Run two-step reconstruction alignment:
        for i in range(0, iterations):   
            print(" iteration %s" %(i+1))
            ## Register all slices to current estimate of volume reconstruction
            # slice_to_volume_registration.run_slice_to_volume_registration(self._HR_volume)

            ## Reconstruct new volume based on updated positions of slices
            volume_reconstruction.update_reconstructed_volume(self._HR_volume)

            self._HR_volume.write(directory=self._dir_results, filename=self._HR_volume_filename+"_"+str(i+1))
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
        self._stack_manager.write(self._dir_results_slices)

        self._HR_volume.write(directory=self._dir_results, filename=self._HR_volume_filename)

        return None