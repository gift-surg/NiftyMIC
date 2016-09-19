## \file IntensityNormalization.py
#  \brief Normalize intensities of slices
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date June 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import itk
import numpy as np
import matplotlib.pyplot as plt

## Import modules from src-folder
import SimpleITKHelper as sitkh
import StackManager as sm
import Stack as st
import Slice as sl


class IntensityNormalization:

    def __init__(self, stacks, target_stack_number):
        self._N_stacks = len(stacks)
        self._target_stack_number = target_stack_number

        ## Number of stacks
        self._N_stacks = len(stacks)

        ## Copy stacks
        self._stacks = [None]*self._N_stacks
        self._stacks_normalized = [None]*self._N_stacks

        for i in range(0, self._N_stacks):
            self._stacks[i] = st.Stack.from_stack(stacks[i])
            self._stacks_normalized[i] = st.Stack.from_stack(stacks[i])

        self._HR_volume_ref = self._stacks[self._target_stack_number].get_isotropically_resampled_stack(interpolator="NearestNeighbor")



    def run_normalization(self):

        ## Create recursive YVV Gaussianfilter
        dim = 3
        image_type = itk.Image[itk.D, dim]
        gaussian_yvv = itk.SmoothingRecursiveYvvGaussianImageFilter[image_type, image_type].New()

        stacks_resampled = [None]*self._N_stacks

        for i in range(0, self._N_stacks):
            stacks_resampled[i] = self._get_resampled_average_stack_of_slices(self._stacks[i])

            ## Debug
            # stacks_resampled[i].show()
            # self._stacks[i].show()

            gaussian_yvv.SetInput(stacks_resampled[i].itk)
            gaussian_yvv.SetSigma(3)
            gaussian_yvv.Update()
            stack_smoothed_itk = gaussian_yvv.GetOutput()
            stack_smoothed_itk.DisconnectPipeline()
            stack_smoothed_sitk = sitkh.convert_itk_to_sitk_image(stack_smoothed_itk)

            stacks_resampled[i] = st.Stack.from_sitk_image(stack_smoothed_sitk,"test_"+str(i))


        x = sitk.GetArrayFromImage(stacks_resampled[0].sitk).flatten()
        y = sitk.GetArrayFromImage(stacks_resampled[1].sitk).flatten()
        
        use_intercept = False

        print("Intensity Normalization:")
        if use_intercept:
            c = np.polyfit(x, y, 1)
            p = np.poly1d(c)
            print c

            A = np.ones((x.size,2))
            A[:,1] = x
            [c0, c1] =  np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(y)
            print("Estimated correction coefficients are [c1, c0] = " + str([c1, c0]))
        
        else:
            c0 = 0
            c1 = x.dot(y)/(x.dot(x))

            print("Estimated correction coefficients are [c1, c0] = " + str([c1, c0]))

            p = np.poly1d((c1,c0))
        
        x_corr = p(x)
        
        ## Plot
        plot_figure = 0
        if plot_figure:
            fig = plt.figure(1)
            fig.clf()
            ax = fig.add_subplot(1,1,1)

            ## Looks quite pretty but not tested here. For other applications 
            ## it took quite long to compute. For less amount of points a 
            ## decent option
            ## Source: http://stackoverflow.com/questions/27156381/python-creating-a-2d-histogram-from-a-numpy-matrix
            # hist, xedges, yedges = np.histogram2d(x, y)
            # xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
            # yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
            # c = hist[xidx, yidx]
            # plt.scatter(x, y, c=c)

            x_int = np.array([x.min(), x.max()])
            ax.plot(x, y, 'ro', label="original")
            ax.plot(x_corr, y, 'gs', label="corrected")
            ax.plot(x_int, p(x_int), 'b-', label="fit")

            legend = ax.legend(loc='center', shadow=False, bbox_to_anchor=(0.5, 1.2), ncol=3)

            ## Show grid
            ax.grid()

            plt.gca().set_aspect('equal', adjustable='box')


            plt.xlabel("Stack 0")
            plt.ylabel("Stack 1")

            plt.draw()
            plt.pause(0.5) ## important! otherwise fig is not shown. Also needs plt.show() at the end of the file to keep figure open


        ## Correct stacks only for first stack
        for i in range(0, 1):
            slices = self._stacks_normalized[i].get_slices()
            N_slices = self._stacks_normalized[i].get_number_of_slices()

            for j in range(0, N_slices):
                slice = slices[j]

                slice.sitk = slice.sitk*c1
                slice.itk = sitkh.convert_sitk_to_itk_image(slice.sitk)



    def get_intensity_corrected_stacks(self):

        return self._stacks_normalized


    def _get_resampled_average_stack_of_slices(self, stack):
        default_pixel_value = 0.0

        ## Define helpers to obtain averaged stack
        shape = sitk.GetArrayFromImage(self._HR_volume_ref.sitk).shape
        array = np.zeros(shape)
        array_mask = np.zeros(shape)
        ind = np.zeros(shape)


        slices = stack.get_slices()
        N_slices = stack.get_number_of_slices()

        for j in range(0, N_slices):
            slice = slices[j]

            ## Resample warped stacks
            slice_sitk =  sitk.Resample(
                slice.sitk,
                self._HR_volume_ref.sitk, 
                sitk.Euler3DTransform(), 
                sitk.sitkBSpline, 
                default_pixel_value,
                slice.sitk.GetPixelIDValue())

            ## Resample warped stack masks
            slice_sitk_mask =  sitk.Resample(
                slice.sitk_mask,
                self._HR_volume_ref.sitk, 
                sitk.Euler3DTransform(), 
                sitk.sitkNearestNeighbor, 
                default_pixel_value,
                slice.sitk_mask.GetPixelIDValue())


            ## Get arrays of resampled warped stack and mask
            array_tmp = sitk.GetArrayFromImage(slice_sitk)
            array_mask_tmp = sitk.GetArrayFromImage(slice_sitk_mask)

            ## Store indices of voxels with non-zero contribution
            ind[np.nonzero(array_tmp)] += 1

            ## Sum intensities of stack and mask
            array += array_tmp
            array_mask += array_mask_tmp
            
        ## Average over the amount of non-zero contributions of the stacks at each index
        ind[ind==0] = 1                 # exclude division by zero
        array = np.divide(array,ind.astype(float))    # elemenwise division

        ## Create (joint) binary mask. Mask represents union of all masks
        array_mask[array_mask>0] = 1

        ## Set pixels of the image not specified by the mask to zero
        # if self._mask_volume_voxels:
        array[array_mask==0] = 0
            
        ## Update HR volume (sitk image)
        stack_average_sitk = sitk.GetImageFromArray(array)
        stack_average_sitk.CopyInformation(self._HR_volume_ref.sitk)

        stack_average_sitk_mask = sitk.GetImageFromArray(array_mask)
        stack_average_sitk_mask.CopyInformation(self._HR_volume_ref.sitk_mask)

        stack_average = st.Stack.from_sitk_image(stack_average_sitk, stack.get_filename()+"_normalized", stack_average_sitk_mask)


        return stack_average


            
