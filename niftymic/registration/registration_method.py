##
# \file registration_method.py
# \brief      Abstract class to define a registration method
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2017
#

# Import libraries
import numpy as np
import SimpleITK as sitk
from abc import ABCMeta, abstractmethod

import pysitk.python_helper as ph

import niftymic.base.stack as st
import niftymic.base.slice as sl


##
# Abstract class for registration methods
# \date       2017-08-09 11:22:51+0100
#
class RegistrationMethod(object):
    __metaclass__ = ABCMeta

    ##
    # Store information for registration methods and initialize additional
    # variables
    # \date       2017-08-09 11:23:14+0100
    #
    # \param      self             The object
    # \param      fixed            Fixed image as Stack/Slice object
    # \param      moving           The moving
    # \param      use_fixed_mask   The use fixed mask
    # \param      use_moving_mask  The use moving mask
    # \param      use_verbose          The use_verbose
    #
    def __init__(self,
                 fixed,
                 moving,
                 use_fixed_mask,
                 use_moving_mask,
                 use_verbose):

        self._fixed = fixed
        self._moving = moving
        self._use_fixed_mask = use_fixed_mask
        self._use_moving_mask = use_moving_mask
        self._use_verbose = use_verbose

        self._computational_time = ph.get_zero_time()
        self._registration_method = None

    ##
    # Sets the fixed image
    # \date       2017-08-08 16:45:45+0100
    #
    # \param      self   The object
    # \param      fixed  The fixed image as Stack/Slice object
    #
    def set_fixed(self, fixed):
        self._fixed = fixed

    ##
    # Gets the fixed image
    # \date       2017-08-08 16:45:58+0100
    #
    # \param      self  The object
    #
    # \return     The fixed image as Stack/Slice object.
    #
    def get_fixed(self):
        return self._fixed

    ##
    # Sets the moving image
    # \date       2017-08-08 16:45:45+0100
    #
    # \param      self    The object
    # \param      moving  The moving image as Stack/Slice object
    #
    #
    def set_moving(self, moving):
        self._moving = moving

    ##
    # Gets the moving image
    # \date       2017-08-08 16:45:58+0100
    #
    # \param      self  The object
    #
    # \return     The moving image as Stack/Slice object.
    #
    def get_moving(self):
        return self._moving

    ##
    # Specify whether fixed mask shall be used for registration
    # \date       2017-08-08 16:48:03+0100
    #
    # \param      self            The object
    # \param      use_fixed_mask  Turn on/off use of fixed mask; bool
    #
    def use_fixed_mask(self, use_fixed_mask):
        self._use_fixed_mask = use_fixed_mask

    ##
    # Specify whether moving mask shall be used for registration
    # \date       2017-08-08 16:48:03+0100
    #
    # \param      self             The object
    # \param      use_moving_mask  Turn on/off use of moving mask; bool
    #
    def use_moving_mask(self, use_moving_mask):
        self._use_moving_mask = use_moving_mask

    ##
    # Sets the use_verbose.
    # \date       2017-08-08 16:50:13+0100
    #
    # \param      self     The object
    # \param      use_verbose  Turn on/off use_verbose output; bool
    #
    def use_verbose(self, use_verbose):
        self._use_verbose = use_verbose

    ##
    # Gets the computational time it took to perform the registration
    # \date       2017-08-08 16:59:45+0100
    #
    # \param      self  The object
    #
    # \return     The computational time.
    #
    def get_computational_time(self):
        return self._computational_time

    ##
    # Gets the obtained registration transform.
    # \date       2017-08-08 16:52:36+0100
    #
    # \param      self  The object
    #
    # \return     The registration transform as sitk object.
    #
    def get_registration_transform_sitk(self):
        return self._registration_transform_sitk

    ##
    # Run the registration method
    # \date       2017-08-08 17:01:01+0100
    #
    # \param      self  The object
    #
    def run_registration(self):

        if not isinstance(self._fixed, st.Stack) and \
                not isinstance(self._fixed, sl.Slice):
            raise TypeError("Fixed image must be of type 'Stack' or 'Slice'")

        if not isinstance(self._moving, st.Stack) and \
                not isinstance(self._moving, sl.Slice):
            raise TypeError("Moving image must be of type 'Stack' or 'Slice'")

        time_start = ph.start_timing()

        # Execute registration method
        self._run_registration()

        # Get computational time
        self._computational_time = ph.stop_timing(time_start)

        if self._use_verbose:
            ph.print_info("Required computational time: %s" %
                          (self.get_computational_time()))

    @abstractmethod
    def _run_registration(self):
        pass

    ##
    # Gets the warped moving image, i.e. moving image warped and resampled to
    # the fixed grid
    # \date       2017-08-08 16:58:30+0100
    #
    # \param      self  The object
    #
    # \return     The warped moving image as Stack/Slice object
    #
    @abstractmethod
    def get_warped_moving(self):
        pass


##
# Abstract class for affine registration methods
# \date       2017-08-09 11:22:51+0100
#
class AffineRegistrationMethod(RegistrationMethod):
    __metaclass__ = ABCMeta

    ##
    # Store information for registration methods and initialize additional
    # variables
    # \date       2017-08-09 11:23:14+0100
    #
    # \param      self             The object
    # \param      fixed            Fixed image as Stack/Slice object
    # \param      moving           The moving
    # \param      use_fixed_mask   The use fixed mask
    # \param      use_moving_mask  The use moving mask
    # \param      use_verbose          The use_verbose
    #
    def __init__(self,
                 fixed,
                 moving,
                 use_fixed_mask,
                 use_moving_mask,
                 use_verbose,
                 registration_type,
                 ):

        RegistrationMethod.__init__(self,
                                    fixed=fixed,
                                    moving=moving,
                                    use_fixed_mask=use_fixed_mask,
                                    use_moving_mask=use_moving_mask,
                                    use_verbose=use_verbose,
                                    )
        self._registration_type = registration_type

    ##
    # Sets the registration type.
    # \date       2017-02-02 16:42:13+0000
    #
    # \param      self               The object
    # \param      registration_type  The registration type
    #
    def set_registration_type(self, registration_type):
        if registration_type not in self._REGISTRATION_TYPES:
            raise ValueError("Possible registration types: " +
                             str(self._REGISTRATION_TYPES))
        self._registration_type = registration_type

    ##
    # Gets the registration type.
    # \date       2017-08-08 19:58:30+0100
    #
    # \param      self  The object
    #
    # \return     The registration type as string.
    #
    def get_registration_type(self):
        return self._registration_type

    ##
    # Gets the warped moving image, i.e. moving image warped and resampled to
    # the fixed grid
    # \date       2017-08-08 16:58:30+0100
    #
    # \param      self  The object
    #
    # \return     The warped moving image as Stack/Slice object
    #
    def get_warped_moving(self):

        warped_moving_sitk_mask = sitk.Resample(
            self._moving.sitk_mask,
            self._fixed.sitk,
            self.get_registration_transform_sitk(),
            sitk.sitkNearestNeighbor,
            0,
            self._moving.sitk_mask.GetPixelIDValue(),
        )

        if isinstance(self._moving, st.Stack):
            warped_moving = st.Stack.from_sitk_image(
                image_sitk=self._get_warped_moving_sitk(),
                filename=self._moving.get_filename(),
                image_sitk_mask=warped_moving_sitk_mask
            )
        else:
            warped_moving = sl.Slice.from_sitk_image(
                image_sitk=self._get_warped_moving_sitk(),
                filename=self._moving.get_filename(),
                image_sitk_mask=warped_moving_sitk_mask
            )
        return warped_moving

    ##
    # Gets the fixed image transformed by the obtained registration transform.
    #
    # The returned image will align the fixed image with the moving image as
    # found during the registration.
    # \date       2017-08-08 16:53:21+0100
    #
    # \param      self  The object
    #
    # \return     The transformed fixed as Stack/Slice object
    #
    def get_transformed_fixed(self):
        fixed = st.Stack.from_stack(self._fixed)
        fixed.update_motion_correction(self.get_registration_transform_sitk())
        return fixed

    @abstractmethod
    def _get_warped_moving_sitk(self):
        pass
