#!/usr/bin/python

## \file StacksVisualizer.py
#  \brief 
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Apr 2017


## Import libraries 
import SimpleITK as sitk
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

## Import modules from src-folder
import utilities.FigureEventHandling as feh
import utilities.FilenameParser as fp


##
# Helper to visualize geometries of a list of stacks
# \date       2017-04-07 15:48:55+0100
#
class StacksVisualizer(object):

    ##
    # Helper class used to visualize 3D arrows
    # \date       2017-04-07 15:44:48+0100
    # \see http://stackoverflow.com/questions/27134567/3d-vectors-in-python
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)


    ##
    # Create instance based on list of filenames
    # \date       2017-04-07 15:45:20+0100
    #
    # \param      cls        The cls
    # \param      dir_input  input directory, string
    # \param      filenames  list of filenames pointing to nifti images without
    #                        filename extension 'nii.gz'
    #
    # \return     class instance
    #
    @classmethod
    def from_filenames(cls, dir_input, filenames):

        stacks_visualizer = cls()

        N_stacks = len(filenames)

        stacks_visualizer._N_stacks = N_stacks
        stacks_visualizer._stacks_sitk = [None] * N_stacks
        stacks_visualizer._titles = [None] * N_stacks

        for i in range(0, N_stacks):
            stacks_visualizer._stacks_sitk[i] = sitk.ReadImage(dir_input + "/" + filenames[i] + ".nii.gz")
            stacks_visualizer._titles[i] = filenames[i]

        return stacks_visualizer


    ##
    # Shows the slice select directions.
    # \date       2017-04-07 15:47:28+0100
    #
    # \param      self    The object
    # \param      colors  The colors as list
    #
    def show_slice_select_directions(self, colors=None):

        if colors is None:
            colors = ['red'] * self._N_stacks

        slice_select_directions = [None] * self._N_stacks

        ## Determine slice-select orientations
        for i in range(0, self._N_stacks):

            stack_sitk = self._stacks_sitk[i]

            ## Extract slice-select direction
            slice_select_directions[i] = stack_sitk[:,:,-1:].GetOrigin() - np.array(stack_sitk[:,:,0:1].GetOrigin())

            ## Normalize direction
            slice_select_directions[i] = slice_select_directions[i] / np.linalg.norm(slice_select_directions[i])

        ## Visualize slice-select orientations
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(0, self._N_stacks):
            arrow = slice_select_directions[i]
            arrow_3D = self.Arrow3D(
                [0,arrow[0]],[0,arrow[1]],[0,arrow[2]], 
                color=colors[i], alpha=0.8, lw=3, arrowstyle="-|>", mutation_scale=20); 
            ax.add_artist(arrow_3D)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_aspect('equal')
        
        step = 0.5
        tick_range_max = 1
        ax.set_xticks(np.arange(-tick_range_max, tick_range_max+step, step))
        ax.set_yticks(np.arange(-tick_range_max, tick_range_max+step, step))
        ax.set_zticks(np.arange(-tick_range_max, tick_range_max+step, step))
        
        plt.show(block=False)

        # dir_output = "/tmp/"
        # filename_output = "foo.pdf"
        # fig.savefig(dir_output + filename_output)