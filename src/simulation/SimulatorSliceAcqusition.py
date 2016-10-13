## \file SimulatorSliceAcqusition.py
#  \brief Based on a given volume, this class aims to simulate the slice
#       acquisition.
#
#       Based on the slice acquisition model,
#                   \f[ y_k = D_k B_k W_k x \f]
#       the orthogonal acquisition of stacks based on their slices \f$ y_k \f$ 
#       are simulated from a volume \f$ x \f$.
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


## Import libraries
import itk
import SimpleITK as sitk
import numpy as np
import os                       # used to execute terminal commands in python
import sys
sys.path.append("../src/")

## Import modules from src-folder
import SimpleITKHelper as sitkh
import Stack as st
import Slice as sl
import PSF as psf


## Pixel type of used 3D ITK image
pixel_type = itk.D

## ITK image type 
image_type = itk.Image[pixel_type, 3]


## Class simulating the slice acquisition
class SliceAcqusition:

    ## Constructor
    #  \param[in] HR_volume Stack object containing the HR volume
    def __init__(self, HR_volume, output_spacing=(1,1,3.85), interpolator_type="OrientedGaussian", alpha_cut=3, motion_type="Random"):
        self._HR_volume = HR_volume
        self._stacks_simulated = []
        self._affine_transforms = []
        self._rigid_motion_transforms = []

        ## Set default standard output image information
        self._output_origin_ref = np.array(HR_volume.sitk.GetOrigin())
        self._output_direction_ref = HR_volume.sitk.GetDirection()
        
        ### Define default output spacing of slices and subsequently defined size of simulated stack
        self._output_spacing = np.array(output_spacing)
        self._output_size = self._get_output_size_based_on_spacing(self._output_spacing)

        ## Define dictionary for different types of available interpolators
        self._get_interpolator = {
            "NearestNeighbor"   :   self._get_interpolator_nearest_neighbor,
            "Linear"            :   self._get_interpolator_linear,
            "OrientedGaussian"  :   self._get_interpolator_oriented_gaussian
        }
        self._interpolator_type = interpolator_type     ## default value

        ## Define dictionary for different types of motion studies
        self._get_rigid_motion_transform = {
            "NoMotion"  :   self._get_motion_transform_no_motion,
            "Random"    :   self._get_motion_transform_random
        }
        self._motion_type = motion_type

        ## Chosen cut-off distance for itkOrientedGaussianInterpolateImageFunction
        self._alpha_cut = alpha_cut


    ## Set cut-off distance for itkOrientedGaussianInterpolateImageFunction
    #  \param[in] alpha_cut scalar value
    def set_alpha_cut(self, alpha_cut):
        self._alpha_cut = alpha_cut


    ## Get cut-off distance
    #  \return scalar value
    def get_alpha_cut(self):
        return self._alpha_cut


    ## Set origin in physical space as reference for coordinate system defining
    #  slice acquisition starting point
    #  \param[in] origin 3D array defining origin in physical space
    def set_origin(self, origin):
        self._output_origin_ref = np.array(origin)


    ## Get origin which defines slice acquisition starting point
    #  \return origin as 3D np.array
    def get_origin(self):
        return self._output_origin_ref


    ## Set direction specifying the coordinate system to define the slice
    #  acquisition directions
    #  \param[in] direction_sitk direction as 9D tuple, obtained via GetDirection() e.g.
    def set_direction(self, direction_sitk):
        self._output_direction_ref = direction_sitk


    ## Get direction which defines slice acquisition directions
    #  \return direction as 9D tuple
    def get_direction(self):
        return self._output_direction_ref


    ## Set spacing for slices, i.e. first two coordinates specify in-plane
    #  and last through-plane dimension
    #  \param[in] spacing 3D array containing spacing information
    #  \post output size of stack is set accordingly but can be changed manually via \p set_size
    def set_spacing(self, spacing):
        self._output_spacing = np.array(spacing).astype("float")
        self._output_size = self._get_output_size_based_on_spacing(self._output_spacing)


    ## Get spacing used for slice simulation
    #  \return spacing as 3D np.array
    def get_spacing(self):
        return self._output_spacing


    ## Set size of data array to be acquired, i.e. specify the amount of voxels
    #  in-plane and through plane for each stack
    #  \param[in] size 3D array specifying data array dimensions
    def set_size(self, size):
        self._output_size = np.array(size).astype("int")


    ## Get size of data array
    #  \return 3D numpy integer array
    def get_size(self):
        return self._output_size


    ## Get simulated stacks of slices
    #  \return list of Stack objects
    def get_simulated_stacks(self):
        if not self._stacks_simulated:
            raise ValueError("Error: No stacks have been simulated yet")

        return self._stacks_simulated


    ## Get affine transforms for each slice of each stack as ground truth
    #  after simulation + the applied motion to get there. The affine 
    #  transforms can be used with
    #     \p get_sitk_image_origin_from_sitk_affine_transform and
    #     \p get_sitk_image_direction_from_sitk_affine_transform
    #  to get actual origin and direction of the image as it was acquired
    #  within the HR volume in the physical space
    #  \see \p get_sitk_image_origin_from_sitk_affine_transform and 
    #       \p get_sitk_image_direction_from_sitk_affine_transform
    #       within SimpleITKHelper
    #  \return list of lists holding sitk.AffineTransform and sitk.Euler3DTransform
    #       objects for each stack of slices
    def get_ground_truth_data(self):
        if not self._stacks_simulated:
            raise ValueError("Error: No stacks have been simulated yet")

        return self._affine_transforms, self._rigid_motion_transforms      


    ## Set type for interpolation used to simulate slices
    #  \param[in] interpolator_type Only 'NearestNeighbor', 'Linear' or 'OrientedGaussian' possible
    def set_interpolator_type(self, interpolator_type):
        if interpolator_type not in ["NearestNeighbor", "Linear", "OrientedGaussian"]:
            raise ValueError("Error: interpolator type can only be 'NearestNeighbor', 'Linear' or 'OrientedGaussian'")
        self._interpolator_type = interpolator_type


    ## Get chosen type of interpolation
    #  \return interpolator type as string
    def get_interpolator_type(self):
        return self._interpolator_type


    ## Set type of applied motion to HR volume before slice acquisition
    #  \param[in] motion_type Only 'NoMotion' or 'Random' possible
    def set_motion_type(self, motion_type):
        if motion_type not in ["NoMotion", "Random"]:
            raise ValueError("Error: motion type can only be 'NoMotion' or 'Random'")
        self._motion_type = motion_type


    ## Get chosen type of motion applied to HR volume before slice acquisition
    #  \return motion type as string
    def get_motion_type(self):
        return self._motion_type


    ## Simulate slice acquisition along first axis based on specified origin and 
    #  coordinate system
    def run_simulation_view_1(self):

        ## Obtain in-plane directions
        dir_in_plane_x   = np.array(self._output_direction_ref)[0::3]
        dir_in_plane_y   = np.array(self._output_direction_ref)[1::3]
        
        ## Obtain slice-select direction
        dir_slice_select = np.array(self._output_direction_ref)[2::3]

        ## Define coordinate system along which the slice acquisition shall be simulated
        output_direction_sitk = np.array([dir_in_plane_x, dir_in_plane_y, dir_slice_select]).transpose().flatten()

        ## Simulate stack acquisition along specified direction 
        self._run_stack_acquisition(output_direction_sitk, title="view_1")


    ## Simulate slice acquisition along second axis based on specified origin and 
    #  coordinate system
    def run_simulation_view_2(self):

        ## Obtain in-plane directions
        dir_in_plane_x   = np.array(self._output_direction_ref)[2::3]
        dir_in_plane_y   = np.array(self._output_direction_ref)[0::3]
        
        ## Obtain slice-select direction
        dir_slice_select = np.array(self._output_direction_ref)[1::3]

        ## Define coordinate system along which the slice acquisition shall be simulated
        output_direction_sitk = np.array([dir_in_plane_x, dir_in_plane_y, dir_slice_select]).transpose().flatten()

        ## Simulate stack acquisition along specified direction 
        self._run_stack_acquisition(output_direction_sitk, title="view_2")


    ## Simulate slice acquisition along third axis based on specified origin and 
    #  coordinate system
    def run_simulation_view_3(self):

        ## Obtain in-plane directions
        dir_in_plane_x   = np.array(self._output_direction_ref)[1::3]
        dir_in_plane_y   = np.array(self._output_direction_ref)[2::3]
        
        ## Obtain slice-select direction
        dir_slice_select = np.array(self._output_direction_ref)[0::3]

        ## Define coordinate system along which the slice acquisition shall be simulated
        output_direction_sitk = np.array([dir_in_plane_x, dir_in_plane_y, dir_slice_select]).transpose().flatten()

        ## Simulate stack acquisition along specified direction 
        self._run_stack_acquisition(output_direction_sitk, title="view_3")


    ## Get output size of stack based on new spacing. It considers
    #  the maximum edge length of the "stack cube". Based on that, all
    #  stacks, acquired in all orthogonal directions, have the same amount
    #  of slices and the target object is fully covered
    #  \param[in] spacing 3D array, defining spacing in x, y and z
    #  return size as 3D integer numpy array
    def _get_output_size_based_on_spacing(self, spacing):
        spacing_HR_volume = np.array(self._HR_volume.sitk.GetSpacing())
        size_HR_volume = np.array(self._HR_volume.sitk.GetSize())
        
        ## Use maximal edge length of stack
        length = np.max(size_HR_volume*spacing_HR_volume)

        return np.round( np.ones(3)*length / spacing ).astype("int")


    ## Run slice acquisition in desired direction based on specified origin
    #  \param[in] output_direction_sitk direction as 9D tuple specifying the slice acquisition
    #  \param[in] title name of the acquisition
    def _run_stack_acquisition(self, output_direction_sitk, title="None"):
        
        ## Obtain slice-select direction
        dir_slice_select = np.array(output_direction_sitk)[2::3]

        ## Create zero array for simulated stack and its mask
        nda = np.zeros(self._output_size[::-1])
        nda_mask = np.zeros(self._output_size[::-1]).astype("uint8")

        ## Initialize Resample Image Filter used to simulate acquisitions
        resampler = itk.ResampleImageFilter[image_type, image_type].New()
        
        resampler.SetDefaultPixelValue( 0.0 )
    
        ## Set output image information
        resampler.SetSize( (self._output_size[0], self._output_size[1], 1) )
        resampler.SetOutputSpacing( self._output_spacing )
        resampler.SetOutputDirection( sitkh.get_itk_direction_form_sitk_direction(output_direction_sitk) )

        ## Prepare to store correct physical positions of acquired slices
        rigid_motion_transforms = []
        affine_transforms = []

        ## Simulate stack acquisition slice by slice
        for i in range(0, self._output_size[2]):

            ## Shift origin along slice-select direction
            origin = self._output_origin_ref + i*self._output_spacing[2]*dir_slice_select 

            ## Update origin based on slice-select direction
            resampler.SetOutputOrigin( tuple(origin) )

            ## Apply motion:
            #  Motion is applied to ref image (slice) before flo image (HR volume)
            #  is resampled
            rigid_motion_transform_itk = self._get_rigid_motion_transform[self._motion_type]()
            resampler.SetTransform( rigid_motion_transform_itk )

            ## Set interpolator
            interpolator = self._get_interpolator[self._interpolator_type](output_direction_sitk, rigid_motion_transform_itk)
            resampler.SetInterpolator(interpolator)
            
            ## Simulate slice
            resampler.SetInput( self._HR_volume.itk )

            resampler.UpdateLargestPossibleRegion()
            resampler.Update()

            slice_itk = resampler.GetOutput()
            slice_itk.DisconnectPipeline()

            ## Simulate mask
            resampler.SetInput( self._HR_volume.itk_mask )

            resampler.UpdateLargestPossibleRegion()
            resampler.Update()

            slice_itk_mask = resampler.GetOutput()
            slice_itk_mask.DisconnectPipeline()
            
            ## Convert itk image and mask to sitk objects for easier handling
            slice_sitk = sitkh.convert_itk_to_sitk_image(slice_itk)
            slice_sitk_mask = sitkh.convert_itk_to_sitk_image(slice_itk_mask)

            ## Fill array information with acquired slice
            nda[i,:,:] = sitk.GetArrayFromImage(slice_sitk)
            nda_mask[i,:,:] = np.clip(np.round(sitk.GetArrayFromImage(slice_sitk_mask)),0,1)
            # nda_mask[i,:,:] = np.floor(sitk.GetArrayFromImage(slice_sitk_mask))

            ## Convert applied motion to sitk format to store ground truth motion transform
            # rigid_motion_transform_sitk = sitkh.get_sitk_AffineTransform_from_itk_AffineTransform(rigid_motion_transform_itk)
            rigid_motion_transform_sitk = sitkh.get_sitk_Euler3DTransform_from_itk_Euler3DTransform(rigid_motion_transform_itk)

            rigid_motion_transforms.append(rigid_motion_transform_sitk)

            ## Store actually acquired position of slice within HR volume in physical space
            affine_transform = self._get_ground_truth_affine_transform(output_direction_sitk, origin, slice_sitk, rigid_motion_transform_sitk)
            affine_transforms.append(affine_transform)

        ## Create stack of simulated slices and associated mask
        stack_simulated_sitk = sitk.GetImageFromArray(nda)
        stack_simulated_sitk.SetOrigin(self._output_origin_ref)
        stack_simulated_sitk.SetDirection(output_direction_sitk)
        stack_simulated_sitk.SetSpacing(self._output_spacing)

        stack_simulated_sitk_mask = sitk.GetImageFromArray(nda_mask)
        stack_simulated_sitk_mask.SetOrigin(self._output_origin_ref)
        stack_simulated_sitk_mask.SetDirection(output_direction_sitk)
        stack_simulated_sitk_mask.SetSpacing(self._output_spacing)

        ## Only slices which contain segmented pixels, i.e. get smallest
        #  rectangular region comprising segmented voxels
        slice_range = self._get_rectangular_masked_region(stack_simulated_sitk_mask)[2]

        ## Create Stack object
        stack = st.Stack.from_sitk_image(stack_simulated_sitk[:,:,slice_range[0]:slice_range[1]], name=title, image_sitk_mask=stack_simulated_sitk_mask[:,:,slice_range[0]:slice_range[1]])
        
        ## Append results
        self._stacks_simulated.append(stack)
        self._affine_transforms.append(affine_transforms[slice_range[0]:slice_range[1]])
        self._rigid_motion_transforms.append(rigid_motion_transforms[slice_range[0]:slice_range[1]])


    ## Get ITK based nearest neighbor interpolator
    #  \param[in] output_direction_sitk dummy variable, not used but important to have same API
    #  \return ITK nearest neighbor interpolator instance
    def _get_interpolator_nearest_neighbor(self, output_direction_sitk, affine_transform_sitk):
        return itk.NearestNeighborInterpolateImageFunction[image_type, pixel_type].New()


    ## Get ITK based linear interpolator
    #  \param[in] output_direction_sitk dummy variable, not used but important to have same API
    #  \return ITK linear interpolator instance
    def _get_interpolator_linear(self, output_direction_sitk, affine_transform_sitk):
        return itk.LinearInterpolateImageFunction[image_type, pixel_type].New()


    ## Get ITK based oriented Gaussian interpolator representing the PSF considering 
    #  the relative position between slice and HR volume
    #  \param[in] output_direction_sitk direction as 9D tuple specifying the slice acquisition direction
    #  \return ITK oriented Gaussian interpolator instance
    def _get_interpolator_oriented_gaussian(self, output_direction_sitk, rigid_motion_transform_itk):
        
        ## Initialize variables
        oriented_gaussian_interpolator = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        PSF = psf.PSF()

        ## Convert applied motion to sitk format
        # motion_transform_sitk = sitkh.get_sitk_AffineTransform_from_itk_AffineTransform(rigid_motion_transform_itk)
        motion_transform_sitk = sitkh.get_sitk_Euler3DTransform_from_itk_Euler3DTransform(rigid_motion_transform_itk)

        ## Get affine transform of slice given output transform (motion not considered and only orientation important, hence dummy origin)
        affine_transform_slice_sitk = sitkh.get_sitk_affine_transform_from_sitk_direction_and_origin(output_direction_sitk, (0,0,0), self._output_spacing)

        ## Get affine transform of simulated slice which will be sampled by considering motion
        output_transformed_affine_sitk = sitkh.get_composite_sitk_affine_transform(motion_transform_sitk, affine_transform_slice_sitk)

        ## Get image direction matrix of affine transform
        output_transformed_direction_sitk = sitkh.get_sitk_image_direction_from_sitk_affine_transform(output_transformed_affine_sitk, self._output_spacing)

        ## Obtain relative oriented Gaussian based on the relative position between (motion corrupted) simulated slice and HR volume
        Cov_HR_coord = PSF.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates_from_direction_and_spacing(output_transformed_direction_sitk, self._output_spacing, self._HR_volume)

        ## Update interpolator
        oriented_gaussian_interpolator.SetAlpha(self._alpha_cut)
        oriented_gaussian_interpolator.SetCovariance(Cov_HR_coord.flatten())

        return oriented_gaussian_interpolator


    ## Get affine transform representing no motion, i.e. identity
    #  \return identity transform as itk.Euler3DTransform object
    def _get_motion_transform_no_motion(self):
        # return itk.AffineTransform[pixel_type, 3].New()
        return itk.Euler3DTransform.New()


    ## Get transform representing random rigid motion
    #  \return random motion transform as itk.Euler3DTransform object
    def _get_motion_transform_random(self):
        angle_deg_max = 5
        translation_max = 5

        # transform = itk.AffineTransform[pixel_type, 3].New()
        transform = itk.Euler3DTransform.New()

        ## Create random translation \f$\in\f$ [\p -translation_max, \p translation_max]
        translation = 2*np.random.rand(3)*translation_max - translation_max

        ## Create random rotation \f$\in\f$ [\p -angle_deg_max, \p angle_deg_max]
        angle_rad_x, angle_rad_y, angle_rad_z = (2*np.random.rand(3)*angle_deg_max - angle_deg_max)/180*np.pi
        
        ## Set resulting rigid motion transform
        transform.SetTranslation( translation )
        transform.SetRotation( angle_rad_x, angle_rad_y, angle_rad_z )

        return transform


    ## Get ground truth affine transform which specifies the actually acquired
    #  position of the slice within the HR volume by taking into account the 
    #  performed HR volume motion.
    #  \param[in] output_direction_sitk direction of slice acquisition based on stationary HR volume position
    #  \param[in] output_origin_sitk origin of slice acquisition based on stationary HR volume position
    #  \param[in] slice_sitk slice as sitk.Image object (required for image spacing)
    #  \param[in] motion_transform_sitk specified motion of HR volume as sitk.AffineTransform object
    #  \return ground truth affine transform which specifies the actually acquired position of the slice within the volume
    def _get_ground_truth_affine_transform(self, output_direction_sitk, output_origin_sitk, slice_sitk, motion_transform_sitk):

        ## Get affine transform of slice for static scenario
        affine_transform_slice = sitkh.get_sitk_affine_transform_from_sitk_direction_and_origin(output_direction_sitk, output_origin_sitk, slice_sitk)

        ## Get ground truth affine transform of slice by taking into account applied motion
        affine_transform = sitkh.get_composite_sitk_affine_transform(motion_transform_sitk, affine_transform_slice)

        return affine_transform


    ## Return rectangular region surrounding masked region. 
    #  \param[in] mask_sitk sitk.Image representing the mask
    #  \param[in] boundary additional boundary surrounding mask in mm (optional). Capped by image domain.
    #  \return range_x pair defining x interval of mask in voxel space 
    #  \return range_y pair defining y interval of mask in voxel space
    #  \return range_z pair defining z interval of mask in voxel space
    def _get_rectangular_masked_region(self, mask_sitk, boundary=0):

        spacing = np.array(mask_sitk.GetSpacing())

        ## Get mask array
        nda = sitk.GetArrayFromImage(mask_sitk)
        
        ## Get shape defining the dimension in each direction
        shape = nda.shape

        ## Set additional offset around identified masked region in voxels
        offset_x = np.round(boundary/spacing[2])
        offset_y = np.round(boundary/spacing[1])
        offset_z = np.round(boundary/spacing[0])

        ## Compute sum of pixels of each slice along specified directions
        sum_xy = np.sum(nda, axis=(0,1)) # sum within x-y-plane
        sum_xz = np.sum(nda, axis=(0,2)) # sum within x-z-plane
        sum_yz = np.sum(nda, axis=(1,2)) # sum within y-z-plane

        ## Find masked regions (non-zero sum!)
        range_x = np.zeros(2)
        range_y = np.zeros(2)
        range_z = np.zeros(2)

        ## Non-zero elements of numpy array nda defining x_range
        ran = np.nonzero(sum_yz)[0]
        range_x[0] = np.max( [0,         ran[0]-offset_x] )
        range_x[1] = np.min( [shape[0], ran[-1]+offset_x+1] )

        ## Non-zero elements of numpy array nda defining y_range
        ran = np.nonzero(sum_xz)[0]
        range_y[0] = np.max( [0,         ran[0]-offset_y] )
        range_y[1] = np.min( [shape[1], ran[-1]+offset_y+1] )

        ## Non-zero elements of numpy array nda defining z_range
        ran = np.nonzero(sum_xy)[0]
        range_z[0] = np.max( [0,         ran[0]-offset_z] )
        range_z[1] = np.min( [shape[2], ran[-1]+offset_z+1] )

        ## Numpy reads the array as z,y,x coordinates! So swap them accordingly
        return range_z.astype(int), range_y.astype(int), range_x.astype(int)

