"""
Figure out how to combine NiftyReg and FLIRT representations with SimpleITK structures
"""

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import unittest
import matplotlib.pyplot as plt

import os                       # used to execute terminal commands in python
import sys
# sys.path.append("../../backups/v1_20150915/")
sys.path.append("../")

# from FileAndImageHelpers import *
# from SimpleITK_PhysicalCoordinates import *

import SimpleITKHelper as sitkh


def plot_compare_sitk_images(image_0, image_1, fig_number=1, flag_continue=0):

    fig = plt.figure(fig_number)
    plt.suptitle("intensity error norm = " + str(np.linalg.norm(sitk.GetArrayFromImage(image_0-image_1))))
    plt.subplot(1,3,1)
    plt.imshow(sitk.GetArrayFromImage(image_0), cmap="Greys_r")
    plt.title("image_0")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(sitk.GetArrayFromImage(image_1), cmap="Greys_r")
    plt.title("image_1")
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow(sitk.GetArrayFromImage(image_0-image_1), cmap="Greys_r")
    plt.title("image_0 - image_1")
    plt.axis('off')

    if flag_continue == 0:
        plt.show()
    else:
        plt.show(block=False)       # does not pause, but needs plt.show() at end 
                                    # of file to be visible


def get_inverse_of_rigid_transform_2D(rigid_transform_2D):
    ## Extract parameters of 2D registration
    angle, translation_x, translation_y = rigid_transform_2D.GetParameters()
    center = rigid_transform_2D.GetFixedParameters()

    ## Create transformation used to align moving -> fixed

    ## Obtain inverse translation
    tmp_trafo = sitk.Euler2DTransform((0,0),-angle,(0,0))
    translation_inv = tmp_trafo.TransformPoint((-translation_x, -translation_y))

    ## inverse = R_inv(x-c) - R_inv(t) + c
    rigid_transform_2D_inv = sitk.Euler2DTransform(center, -angle, translation_inv)

    return rigid_transform_2D_inv


def get_transformed_image(image_init, rigid_transform_2D):
    image = sitk.Image(image_init)
    
    affine_transform = sitkh.get_sitk_affine_transform_from_sitk_image(image)

    transform = sitkh.get_composited_sitk_affine_transform(get_inverse_of_rigid_transform_2D(rigid_transform_2D), affine_transform)

    direction = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(transform, image)
    origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(transform, image)

    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


def get_sitk_registration_transform(fixed, moving):

    ## Registration with SimpleITK:
    # initial_transform = sitk.Euler2DTransform()
    initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkLinear)
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(
        # learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10)

    registration_method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    # registration_method.SetOptimizerAsRegularStepGradientDescent(
    #     learningRate=1, minStep=1, numberOfIterations=100)

    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    ## Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()    
    
    registration_method.SetInitialTransform(initial_transform)

    ## Execute 2D registration
    final_transform_2D_sitk = registration_method.Execute(fixed, moving) 
    print("SimpleITK Image Registration Method:")
    print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return final_transform_2D_sitk


def get_NiftyReg_registration_transform(fixed_sitk, moving_sitk):
    fixed = "fixed"
    moving = "moving"

    res_affine_image = moving + "_warped_NiftyReg"
    res_affine_matrix = "affine_matrix_NiftyReg"

    sitk.WriteImage(fixed_sitk, fixed+".nii.gz")
    sitk.WriteImage(moving_sitk, moving+".nii.gz")

    options = "-voff -rigOnly "
    # options = "-voff -platf Cuda=1 "
        # "-rmask " + fixed_mask + ".nii.gz " + \
        # "-fmask " + moving_mask + ".nii.gz " + \
    cmd = "reg_aladin " + options + \
        "-ref " + fixed + ".nii.gz " + \
        "-flo " + moving + ".nii.gz " + \
        "-res " + res_affine_image + ".nii.gz " + \
        "-aff " + res_affine_matrix + ".txt"
    sys.stdout.write("  Rigid registration (NiftyReg reg_aladin) ... ")
    
    sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
    # print(cmd)
    os.system(cmd)
    print "done"

    ## Read registration matrix
    transform = np.loadtxt(res_affine_matrix+".txt")

    ## Extract information of transform:
    A = transform[0:-2,0:-2]

    ## Negative representation of (x,y)-coordinates compared to nifti-header (cf. SimpleITK_PhysicalCoordinates.py) --> negative sign
    t = -transform[0:-2,-1]

    print transform

    ## Extract angle:
    ## Caution: NiftyReg uses mathematically negative representation for rotation!! --> negative sign
    angle = -np.arccos(A[0,0])

    ## Variant 1
    final_transform_2D_NiftyReg = sitk.AffineTransform(A.flatten(), t)

    ## Variant 2
    # final_transform_2D_NiftyReg = sitk.Euler2DTransform((0,0), angle, t)

    ## Delete files used for NiftyReg
    # cmd = "rm " + \
    #         fixed + ".nii.gz " + \
    #         moving + ".nii.gz " + \
    #         res_affine_image + ".nii.gz " + \
    #         res_affine_matrix + ".txt"
    # os.system(cmd)

    return final_transform_2D_NiftyReg


"""
Why does FLIRT not do anything at all?!
"""
def get_FLIRT_registration_transform(fixed_sitk, moving_sitk):
    fixed = "fixed"
    moving = "moving"

    res_affine_image = moving + "_warped_FLIRT"
    res_affine_matrix = "affine_matrix_FLIRT"

    sitk.WriteImage(fixed_sitk, fixed+".nii.gz")
    sitk.WriteImage(moving_sitk, moving+".nii.gz")

    options = "-2D -v -cost normmi -searchcost normmi -init affine_matrix_FLIRT.txt "
    # options = ""
        # "-refweight " + fixed_mask + ".nii.gz " + \
        # "-inweight " + moving_mask + ".nii.gz " + \
        # "-out " + res_affine_image + ".nii.gz " + \
    cmd = "flirt " + options + \
        "-in " + moving + ".nii.gz " + \
        "-ref " + fixed + ".nii.gz " + \
        "-omat " + res_affine_matrix + ".txt"
    sys.stdout.write("  Rigid registration (FLIRT) ... ")
    
    sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
    print(cmd)
    os.system(cmd)
    print "done"

    ## Read registration matrix
    transform = np.loadtxt(res_affine_matrix+".txt")

    ## Extract information of transform:
    A = transform[0:-2,0:-2]

    """
    Work on that
    ## Negative representation of (x,y)-coordinates compared to nifti-header (cf. SimpleITK_PhysicalCoordinates.py) --> negative sign
    """
    t = -transform[0:-2,-1]

    print transform

    """
    Work on that
    ## Extract angle: 
    """
    angle = np.arccos(A[0,0])

    ## Variant 1
    final_transform_2D_FLIRT = sitk.AffineTransform(A.flatten(), t)

    ## Variant 2
    # final_transform_2D_FLIRT = sitk.Euler2DTransform((0,0), angle, t)

    ## Delete files used for FLIRT
    # cmd = "rm " + \
    #         fixed + ".nii.gz " + \
    #         moving + ".nii.gz " + \
    #         res_affine_image + ".nii.gz " + \
    #         res_affine_matrix + ".txt"
    # os.system(cmd)

    # print final_transform_2D_FLIRT

    return final_transform_2D_FLIRT


"""
Unit Test Class
"""

class TestUM(unittest.TestCase):

    def setUp(self):
        pass


    def test_01_sitk_Resample(self):
        angle = np.pi/3
        translation = (10,-20)
        center = (30,40)
        # center = (0,0)

        ## Load image
        fixed = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)

        ## Generate rigid transformation
        rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)

        ## Resample rigidly transformed image
        moving_resampled = sitk.Resample(fixed, rigid_transform_2D, sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())

        ## Get transformed image in physical space:
        moving_warped = get_transformed_image(fixed, rigid_transform_2D)

        ## Resample rigidly transformed fixed to image space of moving_resampled:
        moving_warped_resampled = sitk.Resample(moving_warped, moving_resampled, sitk.Euler2DTransform(), sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())

        ## Optional: Plot outcome
        # plot_compare_sitk_images(moving_warped_resampled, moving_resampled)

        ## Test alginment
        self.assertEqual(np.around(
            np.linalg.norm( sitk.GetArrayFromImage(moving_warped_resampled - moving_resampled) )
            , decimals = accuracy), 0 )


    def test_02_sitk_Registration(self):
        angle = np.pi/30
        translation = (1,-2)
        center = (30,40)
        # center = (0,0)

        ## Load image
        fixed = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)

        ## Generate rigid transformation
        rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)

        ## Get transformed image to register
        moving = get_transformed_image(fixed, rigid_transform_2D)


        ## Optional: Plot
        # moving_resampled = sitk.Resample(moving, fixed, sitk.Euler2DTransform(), sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())
        # plot_compare_sitk_images(fixed, moving_resampled)


        ## Register with SimpleITK:
        final_transform_2D_sitk = get_sitk_registration_transform(fixed, moving)

        ## Resample result:
        ## Transform fixed into moving space and then resample there to bring image back to fixed space
        warped_sitk_registration = sitk.Resample(moving, fixed, final_transform_2D_sitk, sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())

        ## Optional: Plot outcome
        plot_compare_sitk_images(fixed, warped_sitk_registration)
        
        ## Test alginment
        # dim_x, dim_y = np.array(fixed.GetSize())
        # center_x, center_y = dim_x/2, dim_y/2
        self.assertEqual(np.around(
            np.linalg.norm( sitk.GetArrayFromImage(warped_sitk_registration - fixed) )
            # np.linalg.norm( sitk.GetArrayFromImage(warped_sitk_registration - fixed)
            #     [center_x - dim_x/4 : center_x + dim_x/4, center_y - dim_y/4 : center_y + dim_y/4] )
            , decimals = accuracy), 0 )



    def test_03_NifyReg_Registration(self):
        return None


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
    filename =  "BrainWeb_2D"
    # filename =  "placenta_s"
    # filename =  "kidney_s"
    # filename =  "fetal_brain_a"
    # filename =  "fetal_brain_c"
    # filename =  "fetal_brain_s"

    accuracy = 6 # decimal places for accuracy of unit tests

    """
    Unit tests:
    """
    print("\nUnit tests:\n--------------")
    # unittest.main()


    """
    Playground
    """
    # angle = 0
    angle = np.pi/20
    # translation = (1,20)
    translation = (0,0)
    center = (30,40)
    # center = (0,0)


    ## Load image
    fixed = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)

    ## Generate test transformation:
    rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)

    ## Get rigidly transformed image
    moving = get_transformed_image(fixed, rigid_transform_2D)

    
    """
    Registration algorithms
    """
    ## Rigid Registration SimpleITK:
    # final_transform_2D = get_sitk_registration_transform(fixed, moving)

    ## Rigid Registration NiftyReg:
    # final_transform_2D = get_NiftyReg_registration_transform(fixed, moving)

    ## Rigid Registration FLIRT:
    final_transform_2D = get_FLIRT_registration_transform(fixed, moving)


    """
    Resampling
    """
    warped_registration = sitk.Resample(moving, fixed, final_transform_2D, sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())

    """
    Plot
    """
    ## Optional: Plot
    # moving_resampled = sitk.Resample(moving, fixed, sitk.Euler2DTransform(), sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())
    # plot_compare_sitk_images(fixed, moving_resampled,1,1)

    ## Optional: Plot outcome
    plot_compare_sitk_images(fixed, warped_registration,2)



    print angle*180/np.pi


