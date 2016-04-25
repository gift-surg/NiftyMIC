## \file ReconstructionManager.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import numpy as np

## Import modules from src-folder
import StackManager as sm
import InPlaneRigidRegistration as iprr
import FirstEstimateOfHRVolume as efhrv
import SliceToVolumeRegistration as s2vr
import VolumeReconstruction as vr
import DataPreprocessing as dp
import HierarchicalSliceAlignment as hsa


## This class manages the whole reconstruction pipeline
class ReconstructionManager:

    ## Set up output directories to save reconstruction results
    #  \param[in] dir_output output directory where all results will be stored
    #  \param[in] target_stack_number stack chosen to define space and coordinate system of HR reconstruction, integer (optional)
    def __init__(self, dir_output, target_stack_number=0, recon_name="Reconstruction"):

        self._dir_results = dir_output
        self._HR_volume_filename = "recon_" + recon_name
        
        ## Directory of input data used for reconstruction algorithm
        self._dir_results_input_data = self._dir_results + "input_data/"

        ## Directory to store slices and their computed rigid registration transformations
        self._dir_results_slices = self._dir_results + "slices/"

        ## Optional: Directory of (intermediate) segmentation propagation data:
        self._dir_results_input_data_processed = self._dir_results + "input_data_processed/"

        ## Optional: Directory after all DP steps ready for reconstruction algorthm:
        # self._dir_results_input_data_dp_final = self._dir_results + "input_data_dp_final/"

        ## Create directory if not already existing
        os.system("mkdir -p " + self._dir_results)
        os.system("mkdir -p " + self._dir_results_slices)
        os.system("mkdir -p " + self._dir_results_input_data)
        os.system("mkdir -p " + self._dir_results_input_data_processed)

        ## Delete files in folder
        os.system("rm -rf " + self._dir_results_input_data + "*")
        os.system("rm -rf " + self._dir_results_input_data_processed + "*")
        # os.system("mkdir -p " + self._dir_results_input_data)

        ## Variables containing the respective classes
        self._data_preprocessing = dp.DataPreprocessing(self._dir_results_input_data, self._dir_results_input_data_processed, target_stack_number)
        self._in_plane_rigid_registration = None
        self._stack_manager = None
        self._HR_volume = None

        ## Pre-defined values
        self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate = False
        self._flag_register_stacks_before_initial_volume_estimate = False

        self._target_stack_number = target_stack_number


    ## Read input stacks stored from given directory
    #  \param[in] dir_input_to_copy directory where stacks are stored
    #  \param[in] filenames_to_copy filenames of stacks to be considered in that directory
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    def read_input_stacks(self, dir_input_to_copy, filenames_to_copy, suffix_mask="_mask"):
        N_stacks = len(filenames_to_copy)
        filenames = [str(i) for i in range(0, N_stacks)]

        self._stack_manager = sm.StackManager()

        ## Copy files to self._dir_results_input_data:
        ## (Filenames represent continuous numbering)
        for i in range(0, N_stacks):

            ## 1) Copy images:
            cmd = "cp " + dir_input_to_copy + filenames_to_copy[i] + ".nii.gz " \
                        + self._dir_results_input_data + filenames[i] + ".nii.gz"
            print cmd
            os.system(cmd)

            ## 2) Copy masks:
            if os.path.isfile(dir_input_to_copy + filenames_to_copy[i] + suffix_mask + ".nii.gz"):
                cmd = "cp " + dir_input_to_copy + filenames_to_copy[i] + suffix_mask + ".nii.gz " \
                            + self._dir_results_input_data + filenames[i] + suffix_mask + ".nii.gz"
                os.system(cmd)
            
        print(str(N_stacks) + " stacks were copied to directory " + self._dir_results_input_data)

        ## Data preprocessing:
        self._data_preprocessing.run_preprocessing(filenames, suffix_mask)

        ## Read stacks:
        self._stack_manager.read_input_stacks(self._dir_results_input_data_processed, filenames, suffix_mask)


    ## Compute first estimate of HR volume to initialize reconstruction algortihm
    #  \post \p self._HR_volume is set to first estimate
    def compute_first_estimate_of_HR_volume_from_stacks(self):
        print("\n--- Compute first estimate of HR volume ---")
        
        ## Instantiate object
        first_estimate_of_HR_volume = efhrv.FirstEstimateOfHRVolume(self._stack_manager, self._HR_volume_filename, self._target_stack_number)

        ## Choose reconstruction approach
        # recon_approach = "SDA"
        recon_approach = "Average"
        first_estimate_of_HR_volume.set_reconstruction_approach(recon_approach) #"SDA" or "Average"
        first_estimate_of_HR_volume.set_SDA_approach("Shepard-YVV") # "Shepard-YVV" or "Shepard-Deriche"
        first_estimate_of_HR_volume.set_SDA_sigma(2)

        ## Forward choice of whether or not in-plane registration within each
        #  stack shall be performed before estimation of initial volume
        first_estimate_of_HR_volume.use_in_plane_registration_for_initial_volume_estimate(self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate)
        
        ## Forward choice of whether or not stacks shall be registered to chosen
        #  target stack or not prior the averaging of stacks for the initial volume estimation
        first_estimate_of_HR_volume.register_stacks_before_initial_volume_estimate(self._flag_register_stacks_before_initial_volume_estimate)
        
        ## Perform estimation of initial HR volume
        first_estimate_of_HR_volume.compute_first_estimate_of_HR_volume()

        ## Get estimation
        self._HR_volume = first_estimate_of_HR_volume.get_HR_volume()

        ## Write
        filename = self._HR_volume_filename + "_0_" + recon_approach
        self._HR_volume.write(directory=self._dir_results, filename=filename)
        

    ## Execute two-step reconstruction alignment approach
    #  \param[in] iterations amount of two-step reconstruction alignment steps
    def run_two_step_reconstruction_alignment_approach(self, iterations=5):
        print("\n--- Run two-step reconstruction alignment approach ---")

        ## Instantiate objects
        slice_to_volume_registration = s2vr.SliceToVolumeRegistration(self._stack_manager, self._HR_volume)
        volume_reconstruction = vr.VolumeReconstruction(self._stack_manager, self._HR_volume)

        ## Choose reconstruction approach
        recon_approach = "SDA"
        sigma = 1.5
        volume_reconstruction.set_reconstruction_approach(recon_approach)
        volume_reconstruction.set_SDA_approach("Shepard-YVV") # "Shepard-YVV" or "Shepard-Deriche"
        volume_reconstruction.set_SDA_sigma(sigma)

        self._HR_volume.show(title="HR_recon_iter0")

        ## Run two-step reconstruction alignment:
        for i in range(0, iterations):   
            print("*** iteration %s/%s" %(i+1,iterations))

            sigma = np.max((1,sigma-0.1))
            volume_reconstruction.set_SDA_sigma(sigma)

            ## Register all slices to current estimate of volume reconstruction
            slice_to_volume_registration.run_slice_to_volume_registration()

            ## Reconstruct new volume based on updated positions of slices
            volume_reconstruction.run_reconstruction()

            filename = self._HR_volume_filename + "_" + str(i+1) + "_" + recon_approach
            self._HR_volume.write(directory=self._dir_results, filename=filename)

            self._HR_volume.show(title="HR_recon_iter"+str(i+1))

        ## Final SRR step
        volume_reconstruction.set_SDA_sigma(0.7)

        volume_reconstruction.set_reconstruction_approach("SRR")
        volume_reconstruction.set_SRR_iter_max(20)
        volume_reconstruction.set_SRR_alpha(0.1)
        volume_reconstruction.set_SRR_approach("TK1")       # "TK0" or "TK1"
        volume_reconstruction.set_SRR_DTD_computation_type("Laplace")
        # volume_reconstruction.set_SRR_DTD_computation_type("FiniteDifference")

        volume_reconstruction.run_reconstruction()
        filename = self._HR_volume_filename + "_" + str(iterations) + "_" + recon_approach
        self._HR_volume.write(directory=self._dir_results, filename=filename)


    ## Execute in-plane rigid registration align slices planarly within stack 
    #  to compensate for planarly occured motion. It used
    #  \post Each slice is updated with new affine matrix defining its updated
    #       spatial position
    def run_in_plane_rigid_registration_within_stack(self):
        self._in_plane_rigid_registration = iprr.InPlaneRigidRegistration(self._stack_manager)


    ## \todo Delete that function. It is used only to check results
    def write_resampled_stacks_after_2D_in_plane_registration(self):
        self._in_plane_rigid_registration.write_resampled_stacks(self._dir_results)
        print("Resampled stacks after in-plane registration successfully written to directory %r " % self._dir_results)


    ## Perform in-plane rigid registration within each stack prior the first
    #  estimate of the HR volume via compute_first_estimate_of_HR_volume_from_stacks
    def set_on_in_plane_rigid_registration_before_estimating_initial_volume(self):
        self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate = True


    ## Do not perform in-plane rigid registration within each stack prior the first
    #  estimate of the HR volume via compute_first_estimate_of_HR_volume_from_stacks
    def set_off_in_plane_rigid_registration_before_estimating_initial_volume(self):
        self._flag_use_in_plane_rigid_registration_for_initial_volume_estimate = False


    ## Rigidly register all stacks with chosen target volume prior the estimation
    #  of first HR volume
    def set_on_registration_of_stacks_before_estimating_initial_volume(self):
        self._flag_register_stacks_before_initial_volume_estimate = True


    ## Do not rigidly register all stacks with chosen target volume prior the estimation
    #  of first HR volume
    def set_off_registration_of_stacks_before_estimating_initial_volume(self):
        self._flag_register_stacks_before_initial_volume_estimate = False


    ## Request all stacks with all slices and their (header) information
    #  \return list of Stack objects
    def get_stacks(self):
        return self._stack_manager.get_stacks()


    ## Set current estimate of HR volume by hand
    #  \param[in] HR_volume current estimate of HR volume, instance of Stack
    def set_HR_volume(self, HR_volume):
        self._HR_volume = HR_volume


    ## Get current estimate of HR volume
    #  \return Stack object containing the current HR volume information
    def get_HR_volume(self):

        try: 
            if self._HR_volume is None:
                raise ValueError("Error: HR volume is not estimated/set yet")
            else:
                return self._HR_volume

        except ValueError as err:
            print(err.message)


    ## Write all important results to predefined output directories
    def write_results(self):
        self._stack_manager.write(self._dir_results_slices)

        self._HR_volume.write(directory=self._dir_results, filename=self._HR_volume_filename)

