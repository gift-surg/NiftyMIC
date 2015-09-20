"""
Use SimpleITK to register images in-plane
"""

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import unittest

import sys
sys.path.append("../../backups/v1_20150915/")

from FileAndImageHelpers import *
from SimpleITK_PhysicalCoordinates import *


"""
Set variables
"""
## Specify data
dir_input = "data/"
dir_output = "results/"
filename =  "0"

accuracy = 6 # decimal places for accuracy of unit tests

## Rotation matrix:
# theta = np.pi
# R = np.array([
#     [np.cos(theta), -np.sin(theta), 0],
#     [np.sin(theta), np.cos(theta), 0],
#     [0, 0, 1]
#     ])
R = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]])


"""
Functions
"""
def get_3D_from_2D_rigid_transform(rigid_transform_2D):
    # Get parameters of 2D registration
    angle_z, translation_x, translation_y = rigid_transform_2D.GetParameters()
    center_x, center_y = rigid_transform_2D.GetFixedParameters()
    
    # Expand obtained translation to 3D vector
    translation_3D = (translation_x, translation_y, 0)
    center_3D = (center_x, center_y, 0)

    # Create 3D rigid transform based on 2D
    # rigid_transform_3D = sitk.Euler3DTransform()
    # rigid_transform_3D.SetParameters((0,0, angle_z, translation_3D))
    return sitk.Euler3DTransform(center_3D, 0, 0, angle_z, translation_3D)


def get_3D_in_plane_alignment_transform_from_2D_rigid_transform(rigid_transform_2D, slice_3D):
    rigid_transform_3D = get_3D_from_2D_rigid_transform(rigid_transform_2D)

    A = get_sitk_affine_matrix_from_sitk_image(slice_3D)
    t = get_sitk_affine_translation_from_sitk_image(slice_3D)

    T_IP = sitk.AffineTransform(A,t)
    T_IP_inv = sitk.AffineTransform(T_IP.GetInverse())

    spacing = np.array(slice_3D.GetSpacing())
    S_inv_matrix = np.diag(1/spacing).flatten()
    S_inv = sitk.AffineTransform(S_inv_matrix,(0,0,0))

    # Trafo T = rigid_trafo_3D o T_IP_inv
    T = get_composited_sitk_affine_transform(rigid_transform_3D,T_IP_inv)

    # Trafo T = S_inv o rigid_trafo_3D o T_IP_inv
    # T = get_composited_sitk_affine_transform(S_inv,T)

    # Compute final composition T = T_IP o S_inv o rigid_trafo_3D o T_IP_inv
    return get_composited_sitk_affine_transform(T_IP,T)


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
    def test_in_plane_registration_of_slice_in_2D_space_of_origin(self):
        
        # Set test data
        angle_z = np.pi/2
        translation_x = 20
        translation_y = 30
        center = (10,10,0)
        point = (0,0,0)   #last coordinate must be zero (only one slice!)

        # Create 2D rigid transform
        rigid_transform_2D = sitk.Euler2DTransform()
        rigid_transform_2D.SetParameters((angle_z, translation_x, translation_y))
        rigid_transform_2D.SetFixedParameters((center[0],center[1]))

        # Extend to 3D rigid transform
        rigid_transform_3D = get_3D_from_2D_rigid_transform(rigid_transform_2D)  

        # Create sitk::simple::AffineTransform object from rigid transform 
        A = get_sitk_affine_matrix_from_sitk_image(slice_3D_fixed)
        t = get_sitk_affine_translation_from_sitk_image(slice_3D_fixed)

        T_IP = sitk.AffineTransform(A,t)
        T_IP_inv = sitk.AffineTransform(T_IP.GetInverse())

        T = get_composited_sitk_affine_transform(rigid_transform_3D,T_IP_inv)
        # T = get_composited_sitk_affine_transform(T_IP,T)


        result_2D = np.array(rigid_transform_2D.TransformPoint(slice_2D_fixed.TransformIndexToPhysicalPoint((point[0],point[1]))))
        result_3D = np.array(rigid_transform_3D.TransformPoint(slice_3D_fixed.TransformIndexToPhysicalPoint(point)))

        # print result_2D
        # print result_3D

        self.assertEqual(np.around(
            np.sum(abs(result_2D - result_3D[0:-1]))
            , decimals = accuracy), 0 )


    def test_in_plane_registration_of_slice_in_2D_space_of_arbitrary_point(self):
        
        # Set test data
        angle_z = np.pi/2
        translation_x = 20
        translation_y = 30
        center = (10,10,0)
        point = (10,10,0)   #last coordinate must be zero (only one slice!)

        # Create 2D rigid transform
        rigid_transform_2D = sitk.Euler2DTransform()
        rigid_transform_2D.SetParameters((angle_z, translation_x, translation_y))
        rigid_transform_2D.SetFixedParameters((center[0],center[1]))

        # Extend to 3D rigid transform
        rigid_transform_3D = get_3D_from_2D_rigid_transform(rigid_transform_2D)
        # print rigid_transform_2D  
        # print rigid_transform_3D    

        A = get_sitk_affine_matrix_from_sitk_image(slice_3D_fixed)
        t = get_sitk_affine_translation_from_sitk_image(slice_3D_fixed)

        T_IP = sitk.AffineTransform(A,t)
        T_IP_inv = sitk.AffineTransform(T_IP.GetInverse())

        T = get_composited_sitk_affine_transform(rigid_transform_3D,T_IP_inv)

        result_2D = np.array(rigid_transform_2D.TransformPoint(slice_2D_fixed.TransformIndexToPhysicalPoint((point[0],point[1]))))
        result_3D = np.array(rigid_transform_3D.TransformPoint(slice_3D_fixed.TransformIndexToPhysicalPoint(point)))

        self.assertEqual(np.around(
            np.sum(abs(result_2D - result_3D[0:-1]))
            , decimals = accuracy), 0 )


    def test_in_plane_registration_in_3D_space(self):

        # Set test data  
        angle_z = np.pi
        translation_x = 20
        translation_y = 40

        #(angle_z, 0, 0) works

        rigid_transform_2D = sitk.Euler2DTransform()
        rigid_transform_2D.SetParameters((angle_z,translation_x, translation_y))

        test_slice = slice_3D_fixed

        # Compute final composition T = T_IP o S_inv o rigid_trafo_3D o T_IP_inv
        T = get_3D_in_plane_alignment_transform_from_2D_rigid_transform(rigid_transform_2D, test_slice)

        """
        Test planar alignment of 3D images
        """
        Nx,Ny,Nz = test_slice.GetSize()

        e_0 = (0,0,0)
        e_x = (Nx,0,0)
        e_y = (0,Ny,0)

        a_0 = np.array(test_slice.TransformIndexToPhysicalPoint(e_0))
        a_x = np.array(test_slice.TransformIndexToPhysicalPoint(e_x)) - a_0
        a_y = np.array(test_slice.TransformIndexToPhysicalPoint(e_y)) - a_0
        a_z = np.cross(a_x,a_y)

        b_0 = np.array(T.TransformPoint(test_slice.TransformIndexToPhysicalPoint(e_0)))
        b_x = np.array(T.TransformPoint(test_slice.TransformIndexToPhysicalPoint(e_x))) - b_0
        b_y = np.array(T.TransformPoint(test_slice.TransformIndexToPhysicalPoint(e_y))) - b_0

        t_3D = b_0 - a_0
        
        angle_z, translation_x, translation_y = rigid_transform_2D.GetParameters()
        t_2D = np.array([translation_x, translation_y])

        # Check: a_x-a_y-plane orthogonal to b_x-b_y-plane
        self.assertEqual(np.around(
            abs(a_z.dot(b_x))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            abs(a_z.dot(b_y))
            , decimals = accuracy), 0 )

        # Check: In-plane translation vector (i.e. translation in a_x-b_x-plane)
        self.assertEqual(np.around(
            abs(a_z.dot(t_3D))
            , decimals = accuracy), 0 )

        # Check: Isometric transformation:
        self.assertEqual(np.around(
            abs(np.linalg.norm(a_x) - np.linalg.norm(b_x))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            abs(np.linalg.norm(a_y) - np.linalg.norm(b_y))
            , decimals = accuracy), 0 )

        print("t_2D = " + str(t_2D))
        print("t_3D = " + str(t_3D))
        print("||t_2D|| = " + str(np.linalg.norm(t_2D)))
        print("||t_3D|| = " + str(np.linalg.norm(t_3D)))
        print("||t_2D|| - ||t_3D|| = " + str(
            abs(np.linalg.norm(t_2D) - np.linalg.norm(t_3D))
            ))

        # self.assertEqual(np.around(
        #     abs(np.linalg.norm(t_2D) - np.linalg.norm(t_3D))
        #     , decimals = accuracy), 0 )


"""
Main Function
"""
if __name__ == '__main__':

    """
    Fetch data
    """
    stack = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat32)
    # stack_mask = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)

    N = stack.GetSize()[-1]


    i = 0
    step = 1

    slice_3D_fixed = stack[:,:,i:i+1]
    slice_3D_moving = stack[:,:,i+step:i+step+1]

    slice_2D_fixed = slice_3D_fixed[:,:,0]
    slice_2D_moving = slice_3D_moving[:,:,0]


    flag_rigid_alignment_in_plane = 1

    ## not necessary but kept for futre reference
    if flag_rigid_alignment_in_plane:
        """
        Register slices in-plane:
        """
        initial_transform = sitk.CenteredTransformInitializer(
            slice_2D_fixed, slice_2D_moving, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)

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

        rigid_transform_2D = registration_method.Execute(
            sitk.Cast(slice_2D_fixed, sitk.sitkFloat32), sitk.Cast(slice_2D_moving, sitk.sitkFloat32))

        # warped = sitk.Resample(slice_2D_moving, slice_2D_fixed, rigid_transform_2D, sitk.sitkLinear, 0.0, slice_2D_moving.GetPixelIDValue())


        angle_z, translation_x, translation_y = rigid_transform_2D.GetParameters()
        translation_2D = (translation_x, translation_y)
        print("translation = " +str(translation_2D))
        print("angle_z = " +str(angle_z))



    """
    Unit tests:

    (Essentially all before not important but kept for just-in-case-lookups later on)
    """
    print("\nUnit tests:\n--------------")
    unittest.main()
