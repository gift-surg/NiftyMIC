## \file IntensityCorrection.py
#  \brief Class containing functions to correct for intensities
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Nov 2016


## Import libraries
import sys
import SimpleITK as sitk
import numpy as np
from scipy.optimize import least_squares
import time

##-----------------------------------------------------------------------------
# \brief      Class to correct intensities either within-stack or given a
#             reference
# \date       2016-11-01 20:12:46+0000
#
class IntensityCorrection(object):

    ##-------------------------------------------------------------------------
    # \brief      Constructor
    # \date       2016-11-01 21:57:13+0000
    #
    # \param      self  The object
    #
    def __init__(self):

        self._object_type = "array_self"

        self._run_intensity_correction = {
            "array_self"        : self._run_self_intensity_correction_array,
            "sitk_self"         : self._run_self_intensity_correction_sitk_image,
            "sitk_reference"    : self._run_intensity_correction_with_reference
        }

        self._use_verbose = False


    ##-------------------------------------------------------------------------
    # \brief      Constructor in case data array is given as input
    # \date       2016-11-01 21:57:20+0000
    #
    # \param      cls         The cls
    # \param      nda         3D numpy array
    # \param      percentile  The percentile
    #
    # \return     Instantiated IntensityCorrection object
    #
    @classmethod
    def from_array(cls, nda, percentile=15):

        intensity_correction = cls()

        intensity_correction._object_type = "array_self"
        intensity_correction._nda = nda
        intensity_correction._percentile = percentile

        return intensity_correction


    ##-------------------------------------------------------------------------
    # \brief      Constructor in case sitk.Image(s) is/are given as input
    # \date       2016-11-01 21:58:28+0000
    #
    # \param      cls             The cls
    # \param      image_sitk      The image sitk
    # \param      reference_sitk  The reference sitk
    # \param      percentile      The percentile
    #
    # \return     Instantiated IntensityCorrection object
    #
    @classmethod
    def from_sitk_image(cls, image_sitk, reference_sitk=None, percentile=10):

        intensity_correction = cls()

        intensity_correction._image_sitk = sitk.Image(image_sitk)
        intensity_correction._percentile = percentile

        if reference_sitk is None:
            intensity_correction._refernce_sitk = None
            intensity_correction._object_type = "sitk_self"
        else:
            intensity_correction._refernce_sitk = sitk.Image(reference_sitk)
            intensity_correction._object_type = "sitk_reference"
            

        return intensity_correction


    def use_verbose(self, verbose):
        self._use_verbose = verbose


    def get_intensity_corrected_array(self):
        return np.array(self._nda)


    def get_intensity_corrected_sitk_image(self):
        return sitk.Image(self._image_sitk)


    ##-------------------------------------------------------------------------
    # \brief      Run intensity correction
    # \date       2016-11-01 21:59:02+0000
    #
    # \param      self  The object
    #
    # \post       self._nda or self._image_sitk are updated
    #
    def run_intensity_correction(self):

        self._run_intensity_correction[self._object_type]()


    ##-------------------------------------------------------------------------
    # \brief      Run intensity correction in case array is given
    # \date       2016-11-01 21:59:02+0000
    #
    # \param      self  The object
    #
    # \post       self._nda is updated
    #
    def _run_self_intensity_correction_array(self):

        i0 = np.percentile(self._nda, self._percentile)
        self._nda[np.where(self._nda<i0)] = 0
        self._nda[np.where(self._nda>=i0)] -= i0


    ##-------------------------------------------------------------------------
    # \brief      Run intensity correction in case image_sitk is given
    # \date       2016-11-01 21:59:02+0000
    #
    # \param      self  The object
    #
    # \post       self._image_sitk is updated
    #
    def _run_self_intensity_correction_sitk_image(self):

        ## Extract numpy data array
        self._nda = sitk.GetArrayFromImage(self._image_sitk)
        
        ## Perform intensity correction on data array
        self._run_self_intensity_correction_array()

        ## Convert back to image with correct header
        image_sitk = sitk.GetImageFromArray(self._nda)
        image_sitk.CopyInformation(self._image_sitk)

        self._image_sitk = image_sitk


    ##-------------------------------------------------------------------------
    # \brief      Run intensity correction in case image_sitk and
    #             reference_sitk are given
    # \date       2016-11-01 22:02:28+0000
    #
    # Intensity correction applies a linear model to align the image with the
    # reference, i.e. image*c1 + c0 = reference. Afterwards the lower
    # intensities of the corrected image are thresholded to the given
    # percentile.
    #
    # \param      self  The object
    #
    # \post       self._image_sitk is updated
    #
    def _run_intensity_correction_with_reference(self):

        shape = self._image_sitk.GetSize()

        self._nda  = sitk.GetArrayFromImage(self._image_sitk)
        nda_ref = sitk.GetArrayFromImage(self._refernce_sitk)
        
        A = np.ones((shape[0]*shape[1],2))


        ## 1st round: Get bias and slopes:
        for i in range(0, shape[2]):
            y = nda_ref[i,:,:].flatten()
            A[:,0] = self._nda[i,:,:].flatten()

            B = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose())
            c1, c0 = B.dot(y)

            if self._use_verbose:
                print("Slice %2d/%d: (c1, c0) = (%.3f, %.3f)" %(i, shape[2]-2, c1, c0))

            self._nda[i,:,:] = self._nda[i,:,:]*c1 + c0

        ## Cap lower intensity to given percentile value
        self._run_self_intensity_correction_array()

        ## 2nd round: Get slope after intensities have been offsetted
        for i in range(0, shape[2]):
            y = nda_ref[i,:,:].flatten()
            A = self._nda[i,:,:].flatten()

            B = A/A.dot(A)
            c1 = B.dot(y)

            if self._use_verbose:
                print("Slice %2d/%d: c1 = %.3f" %(i, shape[2]-2, c1))

            self._nda[i,:,:] = self._nda[i,:,:]*c1


        ## Convert back to image with correct header
        image_sitk = sitk.GetImageFromArray(self._nda)
        image_sitk.CopyInformation(self._image_sitk)

        self._image_sitk = image_sitk



    # ##-----------------------------------------------------------------------------
    # # \brief      Correct intensity values based on linear model. Does not really
    # #             work well
    # # \date       2016-09-19 18:54:51+0100
    # #
    # # \param      nda   The nda
    # #
    # # \return     Corrected 3D numpy array
    # #
    # def _run_self_intensity_correction_array_linear_model(nda):
    #     N_slices = nda.shape[0]
    #     plot_figure = 0

    #     # nda_corrected = np.array(nda)
    #     inplane_shape = nda[0,:,:].shape

    #     for j in range(N_slices-2,-1,-1):
    #         i_ref = j+1
    #     # for j in range(1, N_slices):
    #         # i_ref = j-1
    #         x = nda[i_ref,:,:].flatten()
    #         y = nda[j,:,:].flatten()

    #         c0 = 0
    #         c1 = x.dot(y)/(y.dot(y))
    #         print("i = " + str(j) + ": Estimated correction coefficients are [c1, c0] = " + str([c1, c0]))

    #         p = np.poly1d((c1,c0))
            
    #         y_corr = p(y)

    #         nda[j,:,:] = y_corr.reshape(inplane_shape)

    #         if plot_figure:
    #             fig = plt.figure(1)
    #             fig.clf()

    #             x_int = np.array([x.min(), x.max()])

    #             ax = fig.add_subplot(1,2,1)
    #             ax.plot(x, y, 'ro', label="original")
    #             ax.plot(x, y_corr, 'gs', label="corrected")
    #             ax.plot(x_int, p(x_int), 'b-', label="fit")
    #             ax.set_aspect('equal', adjustable='box')
    #             ax.grid()
    #             legend = ax.legend(loc='center', shadow=False, bbox_to_anchor=(0.5, 1.2), ncol=3)
                
    #             ax = fig.add_subplot(1,2,2)
    #             H, xedges, yedges = np.histogram2d(x, y)
    #             ax.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    #             ax.set_aspect('equal', adjustable='box')

    #             plt.xlabel("slice j-1")
    #             plt.ylabel("slice j")

    #             plt.draw()
    #             plt.pause(0.5) ## important! otherwise fig is not shown. Also needs plt.show() at the end of the file to keep figure open
    #             programPause = raw_input("Press the <ENTER> key to continue...")

    #     return nda