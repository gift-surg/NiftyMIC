##
# \file motion_evaluator.py
# \brief      Class to evaluate computed motions
#
# Should help to assess the registration accuracy.
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       January 2018
#


# Import libraries
import os
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.exceptions as exceptions


##
# Class to evaluate computed motions
# \date       2018-01-25 22:53:37+0000
#
class MotionEvaluator(object):

    def __init__(self, transforms_sitk):
        self._transforms_sitk = transforms_sitk
        self._transform_params = None

        self._scale = {
            # convert radiant to degree for angles
            6: np.array([180. / np.pi, 180. / np.pi, 180. / np.pi, 1, 1, 1]),
        }
        self._labels_long = {
            6: ["angle_x [deg]",
                "angle_y [deg]",
                "angle_z [deg]",
                "t_x [mm]",
                "t_y [mm]",
                "t_z [mm]"],
        }
        self._labels_short = {
            6: ["angle_x",
                "angle_y",
                "angle_z",
                "t_x",
                "t_y",
                "t_z"],
        }

    def run(self):

        # # Eliminate center information
        # if self._transforms_sitk[0].GetName() in \
        #         ["Euler2DTransform", "Euler3DTransform"]:
        #     identity = eval("sitk.Euler%dDTransform()"
        # transforms_sitk = [
        #     sitkh.get_composite_sitk_euler_transform()
        # ]

        # Create (#transforms x DOF) numpy array
        self._transform_params = np.zeros((
            len(self._transforms_sitk),
            len(self._transforms_sitk[0].GetParameters())
        ))

        for j in range(self._transform_params.shape[0]):
            self._transform_params[j, :] = \
                self._transforms_sitk[j].GetParameters()

    def display(self, title=None, dir_output=None):
        pd.set_option('display.width', 1000)
        N_trafos, dof = self._transform_params.shape
        if dof == 6:
            params = self._get_scaled_params(self._transform_params)

            # add mean value
            params = np.concatenate(
                (params,
                 np.mean(params, axis=0).reshape(1, -1),
                 np.std(params, axis=0).reshape(1, -1)
                 ))

            cols = self._labels_long[dof]

        else:
            params = self._transform_params
            cols = ["a%d" % (d + 1) for d in range(0, dof)]

        rows = ["Trafo %d" % (d + 1) for d in range(0, N_trafos)]
        rows.append("Mean")
        rows.append("Std")

        df = pd.DataFrame(params, rows, cols)
        print(df)

        if dir_output is not None:
            title = self._replace_string(title)
            filename = "%s.csv" % title
            ph.create_directory(dir_output)
            df.to_csv(os.path.join(dir_output, filename))

    ##
    # Plot figure to show parameter distribution.
    # Only works for 3D rigid transforms for now.
    # \date       2018-01-25 23:30:45+0000
    #
    # \param      self  The object
    #
    def show(self, title=None, dir_output=None):
        params = self._get_scaled_params(self._transform_params)

        N_trafos, dof = self._transform_params.shape

        fig = plt.figure(title)
        fig.clf()

        x = range(1, N_trafos+1)
        ax = plt.subplot(2, 1, 1)
        for i_param in range(0, 3):
            ax.plot(
                x, params[:, i_param],
                marker=ph.MARKERS[i_param],
                color=ph.COLORS_TABLEAU20[i_param*2],
                linestyle=":",
                label=self._labels_short[dof][i_param],
                markerfacecolor="w",
            )
        ax.set_xticks(x)
        plt.ylabel('Rotation [deg]')
        plt.legend(loc="best")

        ax = plt.subplot(2, 1, 2)
        for i_param in range(0, 3):
            ax.plot(
                x, params[:, 3+i_param],
                marker=ph.MARKERS[i_param],
                color=ph.COLORS_TABLEAU20[i_param*2],
                linestyle=":",
                label=self._labels_short[dof][3+i_param],
                markerfacecolor="w",
            )
        ax.set_xticks(x)
        plt.xlabel('Slice')
        plt.ylabel('Translation [mm]')
        plt.legend(loc="best")
        plt.suptitle(title)

        try:
            # Open windows (and also save them) in full screen
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        except:
            pass

        plt.show(block=False)

        if dir_output is not None:
            title = self._replace_string(title)
            filename = "%s.pdf" % title
            ph.save_fig(fig, dir_output, filename)

    def _get_scaled_params(self, transform_params):
        dof = self._transform_params.shape[1]
        return self._transform_params * self._scale[dof]

    def _replace_string(self, string):
        string = re.sub(" ", "_", string)
        string = re.sub(":", "", string)
        string = re.sub("/", "_", string)
        return string
