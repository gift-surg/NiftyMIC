"""
Use SimpleITK to register images in-plane
"""

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import unittest

import sys
sys.path.append("../../backups/v1_20150915/")
sys.path.append("../")

from FileAndImageHelpers import *
from SimpleITK_PhysicalCoordinates import *

import SimpleITKHelper as sitkh


"""
Functions used for SimpleITK illustrations
"""
#callback invoked when the StartEvent happens, sets up our new data
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []


#callback invoked when the EndEvent happens, do cleanup of data and figure
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    #close figure, we don't want to get a duplicate of the plot latter on
    plt.close()


#callback invoked when the IterationEvent happens, update our data and display new figure    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    #clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    #plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    

#callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
#metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


"""
Unit Test Class

Rigid registration transform stored as rigid_transform_2D in main() is not used here.
"""
## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug accuracy, 2015
class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    """
    Test 
    """
    def test_01_transformation_back_to_origin(self):
        ## Fetch data:
        stack = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat64)

        ## Define vectors pointing along the main axis of the image space
        N_x, N_y, N_z = stack.GetSize()
        spacing = np.array(stack.GetSpacing())

        e_0 = (0,0,0)
        e_x = (N_x,0,0)
        e_y = (0,N_y,0)
        e_z = (0,0,N_z)
        e_xyz = (N_x,N_y,N_z)

        ## Extract affine transformation to transform from Image to Physical Space
        T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(stack)

        ## T = T_rotation_inv o T_origin_inv
        T = sitkh.get_3D_transform_to_align_stack_with_physical_coordinate_system(stack)

        ## Test that image axis in physical space now align with physical coordinate system ones
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_0 - T.TransformPoint(stack.TransformIndexToPhysicalPoint(e_0) ))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_x - T.TransformPoint(stack.TransformIndexToPhysicalPoint(e_x) ))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_y - T.TransformPoint(stack.TransformIndexToPhysicalPoint(e_y) ))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_z - T.TransformPoint(stack.TransformIndexToPhysicalPoint(e_z) ))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_xyz - T.TransformPoint(stack.TransformIndexToPhysicalPoint(e_xyz) ))
            , decimals = accuracy), 0 )


        ## Transformation representing the one needed to incorporated in the image header
        T_PI_align = sitkh.get_composited_sitk_affine_transform(T, T_PI)

        ## Extract origin and direction to write into axis aligned stack
        origin_PI_align = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_align,stack)
        direction_PI_align = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_align,stack)

        ## Update image header
        stack.SetDirection(direction_PI_align)
        stack.SetOrigin(origin_PI_align)

        ## Test whether the update of the image header (set origin and direction) lead to the same results
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_0 - stack.TransformIndexToPhysicalPoint(e_0) )
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_x - stack.TransformIndexToPhysicalPoint(e_x) )
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_y - stack.TransformIndexToPhysicalPoint(e_y) )
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_z - stack.TransformIndexToPhysicalPoint(e_z) )
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_xyz - stack.TransformIndexToPhysicalPoint(e_xyz) )
            , decimals = accuracy), 0 )


    def test_02_alignment_with_PI_align_trafo_coordinate_system(self):
        ## Set test data:
        slice_number = 0
        angle = np.pi/3
        translation = (20,30)
        center = (10,15)

        ## Fetch data:
        stack = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat64)

        ## Define vectors pointing along the main axis of the image space
        N_x, N_y, N_z = stack.GetSize()
        
        e_0 = (0,0,0)
        e_x = (N_x,0,0)
        e_y = (0,N_y,0)
        e_xy = (N_x,N_y,0)


        ## Extract affine transformation to transform from Image to Physical Space
        T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(stack)

        ## T = T_rotation_inv o T_origin_inv
        T = sitkh.get_3D_transform_to_align_stack_with_physical_coordinate_system(stack)

        ## T_PI_align = T_rotation_inv o T_origin_inv o T_PI: Trafo to align stack with physical coordinate system
        ## (Hence, T_PI_align(i) = spacing*i)
        T_PI_align = sitkh.get_composited_sitk_affine_transform(T, T_PI)

        ## Extract direction matrix and origin so that slice is oriented according to T_PI_align (i.e. with physical axes)
        origin_PI_align = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_align,stack)
        direction_PI_align = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_align,stack)

        ## Update image header
        """
        After this line: direction = eye(3) and origin = \0. Only spacing kept!
        Hence: Find proper trafo by hand do get this without numerical approximation or set it by hand!
        """
        stack.SetDirection(direction_PI_align)
        stack.SetOrigin(origin_PI_align)

        ## Fetch slices
        slice_3D = stack[:,:,slice_number:slice_number+1]        
        slice_2D = stack[:,:,slice_number]        

        ## Create 2D rigid transform
        rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)

        ## Extend to 3D rigid transform
        rigid_transform_3D = sitkh.get_3D_from_sitk_2D_rigid_transform(rigid_transform_2D)  

        ## Check results
        point = e_0
        res_2D = rigid_transform_2D.TransformPoint(slice_2D.TransformIndexToPhysicalPoint((point[0],point[1])))
        self.assertEqual(np.around(
            np.linalg.norm(
                np.array([res_2D[0],res_2D[1],0]) -
                np.array(rigid_transform_3D.TransformPoint(slice_3D.TransformIndexToPhysicalPoint(point)))
                )
            , decimals = accuracy), 0 )

        point = e_x
        res_2D = rigid_transform_2D.TransformPoint(slice_2D.TransformIndexToPhysicalPoint((point[0],point[1])))
        self.assertEqual(np.around(
            np.linalg.norm(
                np.array([res_2D[0],res_2D[1],0]) -
                np.array(rigid_transform_3D.TransformPoint(slice_3D.TransformIndexToPhysicalPoint(point)))
                )
            , decimals = accuracy), 0 )

        point = e_y
        res_2D = rigid_transform_2D.TransformPoint(slice_2D.TransformIndexToPhysicalPoint((point[0],point[1])))
        self.assertEqual(np.around(
            np.linalg.norm(
                np.array([res_2D[0],res_2D[1],0]) -
                np.array(rigid_transform_3D.TransformPoint(slice_3D.TransformIndexToPhysicalPoint(point)))
                )
            , decimals = accuracy), 0 )
        
        point = e_xy
        res_2D = rigid_transform_2D.TransformPoint(slice_2D.TransformIndexToPhysicalPoint((point[0],point[1])))
        self.assertEqual(np.around(
            np.linalg.norm(
                np.array([res_2D[0],res_2D[1],0]) -
                np.array(rigid_transform_3D.TransformPoint(slice_3D.TransformIndexToPhysicalPoint(point)))
                )
            , decimals = accuracy), 0 )


    def test_03_in_plane_rigid_transformation_in_3D_space(self):
        ## Set test data:
        slice_number = 0
        angle = np.pi/3
        translation = (20,30)
        center = (10,0)

        ## Create 2D rigid transform
        rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)

        ## Fetch data:
        stack = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat64)

        ## Fetch slice which header is to be updated
        slice_3D = stack[:,:,slice_number:slice_number+1]  

        ## Create deep copy for before/after comparison
        slice_3D_original = sitk.Image(slice_3D)

        ## Define vectors pointing along the main axis of the image space
        spacing = np.array(stack.GetSpacing())
        N_x, N_y, N_z = stack.GetSize()
        
        e_0 = (0,0,0)
        e_x = (N_x,0,0)
        e_y = (0,N_y,0)
        e_xy = (N_x,N_y,0)
        e_c = (center[0],center[1],0)/spacing

        ## Transform to physical origin
        T = sitkh.get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D)

        ## Get in-plane rigid transform in 3D space
        T_PI_in_plane_rotation_3D = sitkh.get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D, T, slice_3D)      

        ## Fetch corresponding information for origin and direction
        origin_PI_in_plane_rotation_3D = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_in_plane_rotation_3D, slice_3D)
        direction_PI_in_plane_rotation_3D = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_in_plane_rotation_3D, slice_3D)

        ## Update information at slice_3D
        slice_3D.SetOrigin(origin_PI_in_plane_rotation_3D)
        slice_3D.SetDirection(direction_PI_in_plane_rotation_3D)


        """
        Test planar alignment of 3D images
        """
        a_0 = np.array(slice_3D_original.TransformIndexToPhysicalPoint(e_0))
        a_x = np.array(slice_3D_original.TransformIndexToPhysicalPoint(e_x)) - a_0
        a_y = np.array(slice_3D_original.TransformIndexToPhysicalPoint(e_y)) - a_0
        a_c = np.array(slice_3D_original.TransformContinuousIndexToPhysicalPoint(e_c))
        a_z = np.cross(a_x,a_y)

        b_0 = np.array(slice_3D.TransformIndexToPhysicalPoint(e_0))
        b_x = np.array(slice_3D.TransformIndexToPhysicalPoint(e_x)) - b_0
        b_y = np.array(slice_3D.TransformIndexToPhysicalPoint(e_y)) - b_0
        b_c = np.array(slice_3D.TransformContinuousIndexToPhysicalPoint(e_c)) 

        t_3D = b_c - a_c

        ## Check: Rigid transformation
        ## Not exactly orthogonal!!
        # self.assertEqual(np.around(
        #     abs(a_x.dot(a_y))
        #     , decimals = accuracy), 0 )
        # self.assertEqual(np.around(
        #     abs(b_x.dot(b_y))
        #     , decimals = accuracy), 0 )

        ## Check: a_x-a_y-plane parallel to b_x-b_y-plane
        self.assertEqual(np.around(
            abs(a_z.dot(b_x))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            abs(a_z.dot(b_y))
            , decimals = accuracy), 0 )

        ## Check: In-plane translation vector (i.e. translation in a_x-b_x-plane)
        self.assertEqual(np.around(
            abs(a_z.dot(t_3D))
            , decimals = accuracy), 0 )

        ## Check: Isometric transformation:
        self.assertEqual(np.around(
            abs(np.linalg.norm(a_x) - np.linalg.norm(b_x))
            , decimals = accuracy-1), 0 )
        self.assertEqual(np.around(
            abs(np.linalg.norm(a_y) - np.linalg.norm(b_y))
            , decimals = accuracy-1), 0 )

        ## Check: Translation vectors have the same norm
        # print("t_2D = " + str(t_2D))
        # print("t_3D = " + str(t_3D))
        # print("||t_2D|| = " + str(np.linalg.norm(t_2D)))
        # print("||t_3D|| = " + str(np.linalg.norm(t_3D)))
        # print("||t_2D|| - ||t_3D|| = " + str(
        #     abs(np.linalg.norm(t_2D) - np.linalg.norm(t_3D))
        #     ))
        t_2D = np.array([translation[0], translation[1]])
        self.assertEqual(np.around(
            abs(np.linalg.norm(t_2D) - np.linalg.norm(t_3D))
            , decimals = accuracy), 0 )

        ## Check: Rotation angle correct:
        alpha_3D_x = np.arccos(a_x.dot(b_x)/(np.linalg.norm(a_x)*np.linalg.norm(b_x)))
        alpha_3D_y = np.arccos(a_y.dot(b_y)/(np.linalg.norm(a_y)*np.linalg.norm(b_y)))

        self.assertEqual(np.around(
            abs(alpha_3D_x - angle)
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            abs(alpha_3D_y - angle)
            , decimals = accuracy), 0 )


"""
Main Function
"""
if __name__ == '__main__':

    """
    Set variables
    """
    ## Specify data
    dir_input = "data/"
    dir_output = "results/"
    filename =  "placenta_s"
    # filename =  "kidney_s"
    # filename =  "fetal_brain_a"
    # filename =  "fetal_brain_c"
    # filename =  "fetal_brain_s"

    accuracy = 6 # decimal places for accuracy of unit tests

    """
    Fetch data
    """
    stack = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat64)
    # stack_mask = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)


    """
    In-plane registration (2D)
    """
    flag_rigid_alignment_in_plane = 0

    ## not necessary but kept for future reference
    if flag_rigid_alignment_in_plane:
        i = 0
        step = 1

        # stack_fixed = stack[:,:,i:i+1]
        # stack_moving = stack[:,:,i+step:i+step+1]

        slice_2D_fixed_test = stack[:,:,i]
        slice_2D_moving_test = stack[:,:,i+step]

        """
        Register slices in-plane:
        """
        initial_transform = sitk.CenteredTransformInitializer(
            slice_2D_fixed_test, slice_2D_moving_test, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)

        registration_method = sitk.ImageRegistrationMethod()

        """
        similarity metric settings
        """
        registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5) #set unsigned int radius
        # registration_method.SetMetricAsCorrelation()
        # registration_method.SetMetricAsDemons()
        # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=20, varianceForJointPDFSmoothing=1.5)
        # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        # registration_method.SetMetricAsMeanSquares()

        # registration_method.SetMetricFixedMask(fixed_mask)
        # registration_method.SetMetricMovingMask(moving_mask)
        # registration_method.SetMetricSamplingStrategy(registration_method.NONE)

        registration_method.SetInterpolator(sitk.sitkLinear)

        """
        optimizer settings
        """
        # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10)
        # registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=1, numberOfIterations=100)

        registration_method.SetOptimizerScalesFromPhysicalShift()

        """
        setup for the multi-resolution framework            
        """
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInitialTransform(initial_transform)

        #connect all of the observers so that we can perform plotting during registration
        # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
        # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

        final_transform = registration_method.Execute(
            sitk.Cast(slice_2D_fixed_test, slice_2D_fixed_test.GetPixelIDValue()), sitk.Cast(slice_2D_moving_test, slice_2D_moving_test.GetPixelIDValue()))

        # warped = sitk.Resample(slice_2D_moving_test, slice_2D_fixed_test, final_transform, sitk.sitkLinear, 0.0, slice_2D_moving_test.GetPixelIDValue())


        angle_z, translation_x, translation_y = final_transform.GetParameters()
        translation_2D = (translation_x, translation_y)
        print("translation = " + str(translation_2D))
        print("angle_z = " + str(angle_z))


    fixed = sitk.ReadImage("../.fixed.nii.gz",sitk.sitkFloat64)
    moving = sitk.ReadImage("../.moving.nii.gz",sitk.sitkFloat64)
    warped = sitk.ReadImage("../.moving_warped_NiftyReg.nii.gz",sitk.sitkFloat64)

    spacing = np.array(fixed.GetSpacing())
    dim = np.array(fixed.GetSize())

    fixed_origin = np.array(fixed.GetOrigin())
    moving_origin = np.array(moving.GetOrigin())
    warped_origin = np.array(warped.GetOrigin())

    transform = np.loadtxt("../.affine_matrix_NiftyReg.txt")
    A = transform[0:-2,0:-2]
    t_2D = transform[0:-2,-1]
    angle_z = np.arccos(A[0,0])


    S = np.diag(spacing)
    S_inv = np.diag(1/spacing)

    center_2D = S.dot(dim/2.)

    # affine = sitk.AffineTransform(A.flatten(), t_2D, center_2D)
    affine = sitk.Euler2DTransform(center_2D, -angle_z, t_2D)

    warped_sitk = sitk.Resample(moving, fixed, affine.GetInverse(), sitk.sitkBSpline, 0.0, moving.GetPixelIDValue())

    print("error = " + str(np.linalg.norm(sitk.GetArrayFromImage(warped_sitk-warped))))

    fig = plt.figure(1)
    plt.suptitle(np.linalg.norm(sitk.GetArrayFromImage(warped_sitk-warped)))
    plt.subplot(1,3,1)
    plt.imshow(sitk.GetArrayFromImage(warped_sitk), cmap="Greys_r")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(sitk.GetArrayFromImage(warped), cmap="Greys_r")
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow(sitk.GetArrayFromImage(warped_sitk-warped), cmap="Greys_r")
    plt.axis('off')
    plt.show()


    """
    Unit tests:

    (Essentially all before not important but kept for just-in-case-lookups later on)
    """
    # print("\nUnit tests:\n--------------")
    # unittest.main()
