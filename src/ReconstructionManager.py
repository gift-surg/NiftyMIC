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


## This class manages the whole reconstruction pipeline
class ReconstructionManager:

    ## Set up output directories to save reconstruction results
    #  \param[in] dir_output output directory where all results will be stored
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

        ## Create directory if not already existing
        os.system("mkdir -p " + self._dir_results)
        os.system("mkdir -p " + self._dir_results_input_data)
        os.system("rm -rf " + self._dir_results_input_data + "*")
        os.system("mkdir -p " + self._dir_results_slices)
        # os.system("mkdir -p " + self._dir_results_seg_prop)
        # os.system("mkdir -p " + self._dir_results_input_data)

        ## Variables containing the respective classes
        self._in_plane_rigid_registration = None
        self._stack_manager = None
        self._HR_volume = None

        ## Pre-defined values
        self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate = False
        self._flag_register_stacks_before_initial_volume_estimate = False


    ## Read input stacks stored from given directory
    #  \param[in] dir_input_to_copy directory where stacks are stored
    #  \param[in] filenames_to_copy filenames of stacks to be considered in that directory
    def read_input_stacks(self, dir_input_to_copy, filenames_to_copy):
        N_stacks = len(filenames_to_copy)
        filenames = [str(i) for i in range(0, N_stacks)]

        self._stack_manager = sm.StackManager()

        ## Copy files to self._dir_results_input_data:
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
        self._stack_manager.read_input_stacks(self._dir_results_input_data, filenames)


    ## Compute first estimate of HR volume to initialize reconstruction algortihm
    #  \post \p self._HR_volume is set to first estimate
    def compute_first_estimate_of_HR_volume_from_stacks(self):
        print("\n--- Compute first estimate of HR volume ---")
        
        ## Instantiate object
        first_estimate_of_HR_volume = efhrv.FirstEstimateOfHRVolume(self._stack_manager, self._HR_volume_filename)

        ## Forward choice of whether or not in-plane registration within each
        #  stack shall be performed before estimation of initial volume
        first_estimate_of_HR_volume.use_in_plane_registration_for_initial_volume_estimate(self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate)
        
        ## Forward choice of whether or not stacks shall be registered to chosen
        #  target stack or not prior the averaging of stacks for the initial volume estimation
        first_estimate_of_HR_volume.register_stacks_before_initial_volume_estimate(self._flag_register_stacks_before_initial_volume_estimate)
        
        ## Perform estimation of initial HR volume
        first_estimate_of_HR_volume.compute_first_estimate_of_HR_volume()

        ## Get estimation
        self._HR_volume = first_estimate_of_HR_volume.get_first_estimate_of_HR_volume()
        

    ## Execute two-step reconstruction alignment approach
    #  \param[in] iterations amount of two-step reconstruction alignment steps
    def run_two_step_reconstruction_alignment_approach(self, iterations=1):
        print("\n--- Run two-step reconstruction alignment approach ---")

        ## Instantiate objects
        slice_to_volume_registration = s2vr.SliceToVolumeRegistration(self._stack_manager)
        volume_reconstruction = vr.VolumeReconstruction(self._stack_manager)

        ## Write
        self._HR_volume.write(directory=self._dir_results, filename=self._HR_volume_filename+"_0")

        ## Run two-step reconstruction alignment:
        for i in range(0, iterations):   
            print(" iteration %s" %(i+1))
            ## Register all slices to current estimate of volume reconstruction
            # slice_to_volume_registration.run_slice_to_volume_registration(self._HR_volume)

            ## Reconstruct new volume based on updated positions of slices
            volume_reconstruction.update_reconstructed_volume(self._HR_volume)

            self._HR_volume.write(directory=self._dir_results, filename=self._HR_volume_filename+"_"+str(i+1))


    def run_in_plane_rigid_registration(self):
        self._in_plane_rigid_registration = iprr.InPlaneRigidRegistration(self._stack_manager)


    def write_resampled_stacks_after_2D_in_plane_registration(self):
        self._in_plane_rigid_registration.write_resampled_stacks(self._dir_results)
        print("Resampled stacks after in-plane registration successfully written to directory %r " % self._dir_results)


    def set_on_in_plane_rigid_registration(self):
        self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate = True


    def set_off_in_plane_rigid_registration(self):
        self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate = False


    def set_on_registration_of_stacks_before_estimating_initial_volume(self):
        self._flag_register_stacks_before_initial_volume_estimate = True


    def set_off_registration_of_stacks_before_estimating_initial_volume(self):
        self._flag_register_stacks_before_initial_volume_estimate = False


    def get_stacks(self):
        return self._stack_manager.get_stacks()

    ## 
    #  \param[in] Current guess of HR volume, instance of Stack    
    def set_HR_volume(self, HR_volume):
        self._HR_volume = HR_volume


    def get_HR_volume(self):
        return self._HR_volume


    def write_results(self):
        self._stack_manager.write(self._dir_results_slices)

        self._HR_volume.write(directory=self._dir_results, filename=self._HR_volume_filename)

