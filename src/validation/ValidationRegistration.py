## \file ValidationRegistration.py
#  \brief 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


## Import libraries
import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os                       # used to execute terminal commands in python
import sys
sys.path.append("../src/")

## Import modules from src-folder
import SimpleITKHelper as sitkh
import StackManager as sm
import SliceToVolumeRegistration as s2vr


## Pixel type of used 3D ITK image
pixel_type = itk.D

## ITK image type 
image_type = itk.Image[pixel_type, 3]

## Class used to evaluate registration results
class ValidationRegistration:

    ## Initialize class for scenario of simulation
    # \param[in] HR_volume_ref HR volume considered as ground truth as Stack object
    def __init__(self, HR_volume_ref, stacks_simulated, ground_truth_data):

        self._HR_volume_ref = HR_volume_ref
        self._stack_manager = sm.StackManager.from_stacks(stacks_simulated)
        self._ground_truth_affine_transforms = ground_truth_data[0]
        self._ground_truth_rigid_motion_transforms = ground_truth_data[1]
        self._slice_to_volume_registration = s2vr.SliceToVolumeRegistration(self._stack_manager, self._HR_volume_ref)

        self._N_stacks = self._stack_manager.get_number_of_stacks()

        self._N_slices_total, self._N_begin_new_stack_array = self._stack_manager.get_total_number_of_slices()

        self._dir_results_figures = "results/figures/"
        ## Create directory if not already existing
        os.system("mkdir -p " + self._dir_results_figures)


    ## Run Slice-to-Volume Registration, i.e. register each slice of each stack with
    #  given reference volume
    #  \param[in] iterations number of Slice-to-Volume Registration steps
    #  \param[in] display_info display information of registration results as we go along
    #  \param[in] save_
    def run_slice_to_volume_registration(self, iterations=3, display_info=0, save_figure=0):
        
        for i in range(0, iterations):
            print("\n*** Iteration %s/%s: Slice-to-Volume Registration" %(i+1,iterations))

            ## Run slice to volume registration
            self._slice_to_volume_registration.run_slice_to_volume_registration(display_info=display_info)

            ## Get affine and rigid motion transforms up to iteration i
            affine_transforms, rigid_motion_transforms = self._stack_manager.get_slice_registration_history_of_stacks()

            ## Show result
            self._show_error_rigid_registration_parameters(rigid_motion_transforms, i, save_figure)
            self._show_target_registration_error(affine_transforms, i, save_figure)


    ## Get list of stored Stack objects
    #  \return List of Stack objects
    def get_stacks(self):
        return self._stack_manager.get_stacks()


    ## Show absolute error of all 6 DOF for rigid registration parameters as a plot
    #  \param[in] rigid_motion_transforms associated to Stack objects 
    #       containing the sitk.Euler3DTransform objects
    #  \param[in] iteration only used to indicate iteration on plot
    def _show_error_rigid_registration_parameters(self, rigid_motion_transforms, iteration, save_figure):

        N_cycles = len(rigid_motion_transforms[0][0]) #equal to iteration
                
        print ("New stacks begin at " + str(self._N_begin_new_stack_array))
        ## 6 Parameters for rigid registration
        N_params = 6
        parameters_est = np.zeros((self._N_slices_total, N_params, N_cycles))
        parameters_gd = np.zeros((self._N_slices_total, N_params))

        ind_slice = 0

        ## Loop through stacks
        for i in range(0, self._N_stacks):

            ## Loop through the slices of current stack
            for j in range(0, len(rigid_motion_transforms[i])):

                for k in range(0, N_cycles):
                    ## Get rigid motion transform of k-th cycle of slice j within stack i
                    rigid_motion_transform = rigid_motion_transforms[i][j][k]

                    ## Get (angle_x, angle_y, angle_x, translation_x, translation_y, translation_z)
                    parameters_est[ind_slice, :, k] = rigid_motion_transform.GetParameters()

                parameters_gd[ind_slice,:] = self._ground_truth_rigid_motion_transforms[i][j].GetParameters()
                ind_slice += 1
        

        titles = ["angle_x", "angle_y", "angle_z", "t_x", "t_y", "t_z"]
        factors = (180/np.pi, 180/np.pi, 180/np.pi, 1, 1, 1)
        units = ["deg", "deg", "deg", "mm", "mm", "mm"]
        
        fig = plt.figure(1, figsize=(20.0, 11.0))
        # fig = plt.figure(1)
        fig.clf()
        fig.suptitle("Accuracy of registration up to iteration %s" %(iteration+1))

        ## Plot result for each parameter
        for i_param in range(0, N_params):

            ax = fig.add_subplot(N_params,1,i_param+1)
            plt.ylabel(titles[i_param] + " (" + units[i_param] + ")")
            # plt.title("Accuracy of " + titles[i_param])

            ## Plot each of all cycles whereby initial and final value are marked distinctly
            ax.plot(np.arange(0,self._N_slices_total), factors[i_param]*(parameters_est[:,i_param,0] - parameters_gd[:,i_param]), 'go', mfc='none', label=str(0) )
            for i_cycle in range(1, N_cycles-1):
                ax.plot(np.arange(0,self._N_slices_total), factors[i_param]*(parameters_est[:,i_param,i_cycle] - parameters_gd[:,i_param]), 'rx', label=str(i_cycle) )
            ax.plot(np.arange(0,self._N_slices_total), factors[i_param]*(parameters_est[:,i_param,-1] - parameters_gd[:,i_param]), 'bo', label=str(N_cycles-1) )

            ## Mark first slice of every stack
            # for i_stack in range(0, self._N_stacks):
            #     ax.plot(self._N_begin_new_stack_array[i_stack]*np.ones(2), ax.get_ylim())

            ## Increase xlimit for easier reading
            plt.xlim(-0.5,self._N_slices_total-0.5)

            ## Draw zero level to show zero-error line
            ax.plot(ax.get_xlim(),(0,0),'k', linewidth=2)

            ## Show grid
            ax.grid()

        plt.xlabel("slice")
        ax = fig.add_subplot(N_params,1,1)
        legend = ax.legend(loc='center left', shadow=False, bbox_to_anchor=(0.1, 1.2), ncol=iteration+2)


        # ax.yscale("log")
        # plt.show()
        plt.draw()
        plt.pause(0.5) ## important! otherwise fig is not shown. Also needs plt.show() at the end of the file to keep figure open
        if save_figure:
            fig.savefig(self._dir_results_figures + "validation_registration_parameters_" + str(iteration) + ".eps")


    ## Show absolute error of all 6 DOF for rigid registration parameters as a plot
    #  \param[in] affine_transforms associated to Stack objects 
    #       containing the sitk.AffineTransform objects
    #  \param[in] iteration only used to indicate iteration on plot
    def _show_target_registration_error(self, affine_transforms, iteration, save_figure):
        stacks = self._stack_manager.get_stacks()

        N_cycles = len(affine_transforms[0][0]) #equal to iteration

        RMS_mean = np.zeros((self._N_slices_total, N_cycles))
        RMS_std = np.zeros((self._N_slices_total, N_cycles))

        ind_slice = 0

        for i in range(0, self._N_stacks):
            slices = stacks[i].get_slices()

            for j in range(0, stacks[i].get_number_of_slices()):
                
                ## Get indices of masked pixels, indices \in \R^{dim, N_points}
                indices = np.array(np.where(sitk.GetArrayFromImage(slices[j].sitk_mask)[::-1]==1))

                A_gd = np.array(self._ground_truth_affine_transforms[i][j].GetMatrix()).reshape(3,3)
                t_gd = np.array(self._ground_truth_affine_transforms[i][j].GetTranslation()).reshape(3,1)
                points_gd = A_gd.dot(indices) + t_gd

                for k in range(0, N_cycles):
                    A_est = np.array(affine_transforms[i][j][k].GetMatrix()).reshape(3,3)
                    t_est = np.array(affine_transforms[i][j][k].GetTranslation()).reshape(3,1)
                    points_est = A_est.dot(indices) + t_est

                    RMS_mean[ind_slice,k] = np.sqrt(np.mean(np.sum((points_est - points_gd)**2,0)))
                    RMS_std[ind_slice,k] = np.sqrt(np.std(np.sum((points_est - points_gd)**2,0)))

                ind_slice += 1


        fig = plt.figure(2)
        # fig = plt.figure(2, figsize=(20.0, 11.0))
        fig.clf()
        fig.suptitle("Accuracy of registration up to iteration %s: Root Mean Square error" %(iteration+1))
        ax = fig.add_subplot(1,1,1)

        ## Plot each of all cycles whereby initial and final value are marked distinctly
        for i_cycle in range(0, N_cycles):
            # ax = fig.add_subplot(N_cycles,1,i_cycle+1)
            # plt.title("Cycle %s " %(i_cycle+1))

            if i_cycle is 0:
                ax.plot(np.arange(0,self._N_slices_total), RMS_mean[:,i_cycle], 'go', mfc='none', label=str(i_cycle) )
            elif i_cycle is N_cycles-1:
                ax.errorbar(np.arange(0,self._N_slices_total), RMS_mean[:,i_cycle], yerr=RMS_std[:,-1], fmt=' bo', label=str(i_cycle) )
            else:
                ax.plot(np.arange(0,self._N_slices_total), RMS_mean[:,i_cycle], 'rx', label=str(i_cycle) )
            
            plt.ylabel(" RMS error (mm)")
            ## Increase xlimit for easier reading
            plt.xlim(-0.5,self._N_slices_total-0.5)

        legend = ax.legend(loc='lower left', shadow=False, ncol=iteration+2)
        
        ## Mark first slice of every stack
        # for i_stack in range(0, self._N_stacks):
        #     ax.plot(self._N_begin_new_stack_array[i_stack]*np.ones(2), ax.get_ylim())

        ## Increase xlimit for easier reading
        plt.xlim(-0.5,self._N_slices_total-0.5)

        ## Draw zero level to show zero-error line
        ax.plot(ax.get_xlim(),(0,0),'k', linewidth=2)

        ## Show grid
        ax.grid()

        plt.yscale("log")
        plt.xlabel("slice")
        # plt.show()
        plt.draw()
        plt.pause(0.5) ## important! otherwise fig is not shown. Also needs plt.show() at the end of the file to keep figure open
        if save_figure:
            fig.savefig(self._dir_results_figures + "validation_registration_TRE_" + str(iteration) + ".eps")
