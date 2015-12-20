"""
Figure out how to combine NiftyReg and FLIRT representations with SimpleITK structures
"""

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import unittest

import os                       # used to execute terminal commands in python
import sys
sys.path.append("../")

# from scipy import ndimage       # For computation of center of mass for FLIRT coordinate system conversion
import commands

# from FileAndImageHelpers import *
# from SimpleITK_PhysicalCoordinates import *

## Import modules from src-folder
import SimpleITKHelper as sitkh



"""
Functions
"""
def get_transformed_image(image_init, rigid_transform_2D):
    image = sitk.Image(image_init)
    
    affine_transform = sitkh.get_sitk_affine_transform_from_sitk_image(image)

    transform = sitkh.get_composited_sitk_affine_transform(sitkh.get_inverse_of_sitk_rigid_registration_transform(rigid_transform_2D), affine_transform)

    direction = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(transform, image)
    origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(transform, image)

    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


def get_sitk_rigid_registration_transform_2D(fixed_2D, moving_2D):

    ## Instantiate interface method to the modular ITKv4 registration framework
    registration_method = sitk.ImageRegistrationMethod()

    ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
    initial_transform = sitk.CenteredTransformInitializer(fixed_2D, moving_2D, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    # initial_transform = sitk.Euler2DTransform()

    ## Set the initial transform and parameters to optimize
    registration_method.SetInitialTransform(initial_transform)

    ## Set an image masks in order to restrict the sampled points for the metric
    # registration_method.SetMetricFixedMask(fixed_2D_mask)
    # registration_method.SetMetricMovingMask(moving_2D_mask)

    ## Set percentage of pixels sampled for metric evaluation
    # registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    ## Set interpolator to use
    registration_method.SetInterpolator(sitk.sitkLinear)

    """
    similarity metric settings
    """
    ## Use normalized cross correlation using a small neighborhood for each voxel between two images, with speed optimizations for dense registration
    # registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5)
    
    ## Use negative normalized cross correlation image metric
    # registration_method.SetMetricAsCorrelation()

    ## Use demons image metric
    # registration_method.SetMetricAsDemons(intensityDifferenceThreshold=1e-3)

    ## Use mutual information between two images
    # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=50, varianceForJointPDFSmoothing=3)
    
    ## Use the mutual information between two images to be registered using the method of Mattes2001
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    ## Use negative means squares image metric
    registration_method.SetMetricAsMeanSquares()
    
    """
    optimizer settings
    """
    ## Set optimizer to Nelder-Mead downhill simplex algorithm
    # registration_method.SetOptimizerAsAmoeba(simplexDelta=0.1, numberOfIterations=100, parametersConvergenceTolerance=1e-8, functionConvergenceTolerance=1e-4, withStarts=false)

    ## Conjugate gradient descent optimizer with a golden section line search for nonlinear optimization
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10)

    ## Set the optimizer to sample the metric at regular steps
    # registration_method.SetOptimizerAsExhaustive(numberOfSteps=50, stepLength=1.0)

    ## Gradient descent optimizer with a golden section line search
    # registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    ## Limited memory Broyden Fletcher Goldfarb Shannon minimization with simple bounds
    # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=500, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=200, costFunctionConvergenceFactor=1e+7)

    ## Regular Step Gradient descent optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=1, numberOfIterations=100)

    ## Estimating scales of transform parameters a step sizes, from the maximum voxel shift in physical space caused by a parameter change
    ## (Many more possibilities to estimate scales)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    """
    setup for the multi-resolution framework            
    """
    ## Set the shrink factors for each level where each level has the same shrink factor for each dimension
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])

    ## Set the sigmas of Gaussian used for smoothing at each level
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])

    ## Enable the smoothing sigmas for each level in physical units (default) or in terms of voxels (then *UnitsOff instead)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    ## Connect all of the observers so that we can perform plotting during registration
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    # print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
    # print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    # print("\n")

    ## Execute 2D registration
    final_transform_2D_sitk = registration_method.Execute(fixed_2D, moving_2D) 
    print("SimpleITK Image Registration Method:")
    print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return final_transform_2D_sitk


def get_NiftyReg_rigid_registration_transform_2D(fixed_sitk, moving_sitk):
    fixed = "fixed"
    moving = "moving"

    res_affine_image = moving + "_warped_NiftyReg"
    res_affine_matrix = "affine_matrix_NiftyReg"

    ## Save images prior the use of NiftyReg
    dir_tmp = "tmp/" 
    os.system("mkdir -p " + dir_tmp)

    sitk.WriteImage(fixed_sitk, dir_tmp + fixed+".nii.gz")
    sitk.WriteImage(moving_sitk, dir_tmp + moving+".nii.gz")

    options = "-voff -rigOnly "
    # options = "-voff -platf Cuda=1 "
        # "-rmask " + fixed_mask + ".nii.gz " + \
        # "-fmask " + moving_mask + ".nii.gz " + \
    cmd = "reg_aladin " + options + \
        "-ref " + dir_tmp + fixed + ".nii.gz " + \
        "-flo " + dir_tmp + moving + ".nii.gz " + \
        "-res " + dir_tmp + res_affine_image + ".nii.gz " + \
        "-aff " + dir_tmp + res_affine_matrix + ".txt"
    print(cmd)
    
    sys.stdout.write("  Rigid registration (NiftyReg reg_aladin) ... ")
    sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
    os.system(cmd)
    print "done"

    ## Read registration matrix
    transform = np.loadtxt(dir_tmp+res_affine_matrix+".txt")

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

    ## Delete tmp-folder
    cmd = "rm -rf " + dir_tmp
    os.system(cmd)

    return final_transform_2D_NiftyReg


def get_NiftyReg_rigid_registration_transform_3D(fixed_sitk, moving_sitk):
    fixed = "fixed"
    moving = "moving"

    res_affine_image = moving + "_warped_NiftyReg"
    res_affine_matrix = "affine_matrix_NiftyReg"

    ## Create folder if not existing
    dir_tmp = "tmp/" 
    os.system("mkdir -p " + dir_tmp)

    ## Delete possibly pre-existing files 
    # cmd = "rm -rf " + dir_tmp
    cmd = "rm -f " + dir_tmp + "*"
    # print cmd
    os.system(cmd)

    ## Save images prior the use of NiftyReg
    sitk.WriteImage(fixed_sitk, dir_tmp + fixed + ".nii.gz")
    sitk.WriteImage(moving_sitk, dir_tmp + moving + ".nii.gz")

    options = "-voff -rigOnly -nac "
    # options = "-rigOnly -nac "
    # options = "-voff -platf Cuda=1 "
        # "-rmask " + fixed_mask + ".nii.gz " + \
        # "-fmask " + moving_mask + ".nii.gz " + \
    cmd = "reg_aladin " + options + \
        "-ref " + dir_tmp + fixed + ".nii.gz " + \
        "-flo " + dir_tmp + moving + ".nii.gz " + \
        "-res " + dir_tmp + res_affine_image + ".nii.gz " + \
        "-aff " + dir_tmp + res_affine_matrix + ".txt"
    print(cmd)
    
    sys.stdout.write("  Rigid registration (NiftyReg reg_aladin) ... ")
    sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
    os.system(cmd)
    print "done"

    ## Read registration matrix
    transform = np.loadtxt(dir_tmp+res_affine_matrix+".txt")

    # print transform

    ## Extract information of transform:

    R = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])

    A = transform[0:-1,0:-1]
    t = transform[0:-1,-1]

    # t = R.dot(t)

    # A = np.linalg.inv(A)
    A = R.dot(A).dot(R)
    # A[0,1] *= -1
    # A[1,0] *= -1
    t = R.dot(t)

    print A
    print t

    ## Create SimpleITK transformation
    final_transform_3D_NiftyReg = sitk.AffineTransform(A.flatten(), t)

    ## Delete tmp-folder
    # cmd = "rm -rf " + dir_tmp
    # cmd = "rm -f " + dir_tmp + "*.txt"
    # os.system(cmd)

    return final_transform_3D_NiftyReg


"""
Why does FLIRT not do anything at all?!
"""
def get_FLIRT_rigid_registration_transform_2D(fixed_sitk, moving_sitk):
    fixed = "fixed"
    moving = "moving"

    res_affine_image = moving + "_warped_FLIRT"
    res_affine_matrix = "affine_matrix_FLIRT"


    ## Create folder if not existing
    dir_tmp = "tmp/" 
    os.system("mkdir -p " + dir_tmp)

    ## Delete possibly pre-existing files 
    # cmd = "rm -rf " + dir_tmp
    cmd = "rm -f " + dir_tmp + "*"
    # print cmd
    os.system(cmd)

    ## Save images prior to ethe use of FLIRT
    sitk.WriteImage(fixed_sitk, dir_tmp+fixed+".nii.gz")
    sitk.WriteImage(moving_sitk, dir_tmp+moving+".nii.gz")

    options = "-2D "
    # options = ""
        # "-refweight " + fixed_mask + ".nii.gz " + \
        # "-inweight " + moving_mask + ".nii.gz " + \
    cmd = "flirt " + options + \
        "-in " + dir_tmp + moving + ".nii.gz " + \
        "-ref " + dir_tmp + fixed + ".nii.gz " + \
        "-out " + dir_tmp + res_affine_image + ".nii.gz " + \
        "-omat " + dir_tmp + res_affine_matrix + ".txt"
    print(cmd)

    sys.stdout.write("  Rigid registration (FLIRT) ... ")
    sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
    os.system(cmd)
    print "done"

    ## Read registration matrix
    transform = np.loadtxt(dir_tmp+res_affine_matrix+".txt")

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

    ## Delete tmp-folder
    cmd = "rm -rf " + dir_tmp
    os.system(cmd)

    return final_transform_2D_FLIRT


def get_FLIRT_rigid_registration_transform_3D(fixed_sitk, moving_sitk):
    fixed = "fixed"
    moving = "moving"

    res_affine_image = moving + "_warped_FLIRT"
    res_affine_matrix = "affine_matrix_FLIRT"

    ## Create folder if not existing
    dir_tmp = "tmp/" 
    os.system("mkdir -p " + dir_tmp)

    ## Delete possibly pre-existing files 
    # cmd = "rm -rf " + dir_tmp
    cmd = "rm -f " + dir_tmp + "*"
    # print cmd
    os.system(cmd)

    ## Save images prior the use of FLIRT
    sitk.WriteImage(fixed_sitk, dir_tmp+fixed+".nii.gz")
    sitk.WriteImage(moving_sitk, dir_tmp+moving+".nii.gz")

    # options = "-2D "
    options = ""
        # "-refweight " + fixed_mask + ".nii.gz " + \
        # "-inweight " + moving_mask + ".nii.gz " + \
    cmd = "flirt " + options + \
        "-in " + dir_tmp + moving + ".nii.gz " + \
        "-ref " + dir_tmp + fixed + ".nii.gz " + \
        "-out " + dir_tmp + res_affine_image + ".nii.gz " + \
        "-omat " + dir_tmp + res_affine_matrix + ".txt"
    print(cmd)

    sys.stdout.write("  Rigid registration (FLIRT) ... ")
    sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
    os.system(cmd)
    print "done"

    ## IRTK: Convert a transformation from the FSL flirt format to DOF format
    cmd = "flirt2dof " + \
        dir_tmp + res_affine_matrix + ".txt " + \
        dir_tmp + fixed + ".nii.gz " + \
        dir_tmp + moving + ".nii.gz " + \
        dir_tmp + res_affine_matrix + ".dof"
    print(cmd)
    os.system(cmd)

    ## IRTK: Convert a transformation represented by the file [doffile] to the IRTK project matrix format
    cmd = "dof2mat " + \
        dir_tmp + res_affine_matrix + ".dof " + \
        "-matout " + dir_tmp + res_affine_matrix + "_mat.txt" 
    print(cmd)
    os.system(cmd)
    



    ## Read registration matrix
    # transform = np.loadtxt(dir_tmp+res_affine_matrix+".txt")
    transform = np.loadtxt(dir_tmp+res_affine_matrix+"_mat.txt")

    warped_sitk = sitk.ReadImage(dir_tmp+res_affine_image+".nii.gz")

    print transform

    ## Extract information of transform:
    R = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])


    e_0 = (0,0,0)
    e_x = (1,0,0)
    e_y = (0,1,0)
    e_z = (0,0,1)

    a_0 = np.array(fixed_sitk.TransformIndexToPhysicalPoint(e_0))
    a_x = fixed_sitk.TransformIndexToPhysicalPoint(e_x) - a_0
    a_y = fixed_sitk.TransformIndexToPhysicalPoint(e_y) - a_0
    a_z = fixed_sitk.TransformIndexToPhysicalPoint(e_z) - a_0
    a_x = a_x/np.linalg.norm(a_x)
    a_y = a_y/np.linalg.norm(a_y)
    a_z = a_z/np.linalg.norm(a_z)

    b_0 = R.dot(get_coordinates(dir_tmp, fixed, moving, res_affine_matrix, e_0))
    b_x = R.dot(get_coordinates(dir_tmp, fixed, moving, res_affine_matrix, e_x)) - b_0
    b_y = R.dot(get_coordinates(dir_tmp, fixed, moving, res_affine_matrix, e_y)) - b_0
    b_z = R.dot(get_coordinates(dir_tmp, fixed, moving, res_affine_matrix, e_z)) - b_0
    b_x = b_x/np.linalg.norm(b_x)
    b_y = b_y/np.linalg.norm(b_y)
    b_z = b_z/np.linalg.norm(b_z)

    B0 = np.array(fixed_sitk.GetDirection()).reshape(3,3)
    B0_inv = np.linalg.inv(B0)

    B1 = np.array([b_x,b_y,b_z])
    B1 = B1.transpose()
    # print B1

    print("a_0 = " + str(a_0))
    print("a_x = " + str(a_x))
    print("a_y = " + str(a_y))
    print("a_z = " + str(a_z))

    print("b_0 = " + str(b_0))
    print("b_x = " + str(B1[:,0]))
    print("b_y = " + str(B1[:,1]))
    print("b_z = " + str(B1[:,2]))


    D = B0_inv.dot(B1)

    # print D
    # print np.linalg.det(D)


    center = (0,0,0)
    angle_x = 0
    angle_y = 0
    angle_z = 0



    translation = D.dot(R.dot(b_0)-a_0)

    # A = sitk.Euler3DTransform(center, angle_x, angle_y, angle_z, translation)
    A = sitk.AffineTransform(D.flatten(), translation)

    # sitkh.print_rigid_transformation(A)

    final_transform_3D_FLIRT = A

    # cmd = "echo " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " | " + \
    #     "img2stdcoord " + \
    #     "-std " + dir_tmp + fixed + ".nii.gz " + \
    #     "-img " + dir_tmp + moving + ".nii.gz " + \
    #     "-xfm " + dir_tmp + res_affine_matrix + ".txt - "
    # # print cmd
    # # os.system(cmd)
    # origin = commands.getstatusoutput(cmd)[1]


    # T1 = sitk.AffineTransform(np.zeros((3,3)).flatten(), -center_of_mass_fixed)
    # T2 = sitk.AffineTransform(A.flatten(), np.zeros(3))
    # T3 = sitk.AffineTransform(np.zeros((3,3)).flatten(), center_of_mass_fixed)
    # T3 = sitk.AffineTransform(np.zeros((3,3)).flatten(), center_of_mass_fixed)

    # T = sitkh.get_composited_sitk_affine_transform(T2,T1)
    # T = sitkh.get_composited_sitk_affine_transform(T3,T)


    ## Create SimpleITK transformation
    # final_transform_3D_FLIRT = sitk.AffineTransform(A.flatten(), t)
    # final_transform_3D_FLIRT = T

    ## Delete tmp-folder
    # cmd = "rm -rf " + dir_tmp
    # cmd = "rm -f " + dir_tmp + "*.txt"
    # os.system(cmd)

    return final_transform_3D_FLIRT


def get_coordinates(dir_tmp, fixed, moving, res_affine_matrix, p):
    # http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide

    res_affine_matrix_inv = res_affine_matrix + "_inv"
    cmd = "convert_xfm " + \
        "-omat " + dir_tmp + res_affine_matrix_inv + ".txt " + \
        "-inverse " + dir_tmp + res_affine_matrix + ".txt "
    # print cmd
    os.system(cmd)


    cmd = "echo " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " | " + \
        "img2stdcoord " + \
        "-img " + dir_tmp + fixed + ".nii.gz " + \
        "-std " + dir_tmp + moving + ".nii.gz " + \
        "-xfm " + dir_tmp + res_affine_matrix + ".txt - "

    # cmd = "echo " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " | " + \
    #     "img2stdcoord " + \
    #     "-img " + dir_tmp + moving + ".nii.gz " + \
    #     "-std " + dir_tmp + fixed + ".nii.gz " + \
    #     "-xfm " + dir_tmp + res_affine_matrix_inv + ".txt - "
    
    # print cmd
    transformed_p_str = commands.getstatusoutput(cmd)[1]
    transformed_p_str = transformed_p_str.split(" ")

    ## return transformed point in SimpleITK coordinate system
    transformed_p = np.array((
        float(transformed_p_str[0]), 
        float(transformed_p_str[2]), 
        float(transformed_p_str[4])))

    print transformed_p

    return transformed_p

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
        fixed = sitk.ReadImage(dir_input + filename_2D + ".nii.gz", sitk.sitkFloat64)

        ## Generate rigid transformation
        rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)

        ## Resample rigidly transformed image
        moving_resampled = sitk.Resample(fixed, rigid_transform_2D, sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())

        ## Get transformed image in physical space:
        moving_warped = get_transformed_image(fixed, rigid_transform_2D)

        ## Resample rigidly transformed fixed to image space of moving_resampled:
        moving_warped_resampled = sitk.Resample(moving_warped, moving_resampled, sitk.Euler2DTransform(), sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())

        ## Optional: Plot outcome
        # sitkh.plot_compare_sitk_2D_images(moving_warped_resampled, moving_resampled)

        ## Test whether resampling directly the image via sitk.Resample with provided rigid_transform_2d yields the
        ## same result as transforming first and then resampling with provided identity transform:
        self.assertEqual(np.around(
            np.linalg.norm( sitk.GetArrayFromImage(moving_warped_resampled - moving_resampled) )
            , decimals = accuracy), 0 )


    def test_02_sitk_Registration(self):
        angle = np.pi/30
        translation = (1,-2)
        center = (30,40)
        # center = (0,0)

        ## Load image
        fixed = sitk.ReadImage(dir_input + filename_2D + ".nii.gz", sitk.sitkFloat64)

        ## Generate rigid transformation
        rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)

        ## Get transformed image to register
        moving = get_transformed_image(fixed, rigid_transform_2D)


        ## Optional: Plot
        # moving_resampled = sitk.Resample(moving, fixed, sitk.Euler2DTransform(), sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())
        # sitkh.plot_compare_sitk_2D_images(fixed, moving_resampled)


        ## Register with SimpleITK:
        final_transform_2D_sitk = get_sitk_rigid_registration_transform_2D(fixed, moving)

        ## Resample result:
        ## Transform fixed into moving space and then resample there to bring image back to fixed space
        warped_sitk_registration = sitk.Resample(moving, fixed, final_transform_2D_sitk, sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())
       
        
        ## Test alginment
        # dim_x, dim_y = np.array(fixed.GetSize())
        # center_x, center_y = dim_x/2, dim_y/2
        try:
            self.assertEqual(np.around(
                np.linalg.norm( sitk.GetArrayFromImage(warped_sitk_registration - fixed) )
                # np.linalg.norm( sitk.GetArrayFromImage(warped_sitk_registration - fixed)
                #     [center_x - dim_x/4 : center_x + dim_x/4, center_y - dim_y/4 : center_y + dim_y/4] )
                , decimals = accuracy), 0 )

        except Exception as e:
            print(self.id() + " failed and image is plotted for further investigation")
            ## Plot outcome
            sitkh.plot_compare_sitk_2D_images(fixed, warped_sitk_registration)
            # self.skipTest(MyTestCase)


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
    filename_2D =  "BrainWeb_2D"
    # filename_3D =  "placenta_s"
    # filename_3D =  "kidney_s"
    # filename_3D =  "fetal_brain_a"
    # filename_3D =  "fetal_brain_c"
    filename_3D =  "fetal_brain_s"
    # filename_3D =  "fetal_brain_s_origin0"
    # filename_3D =  "fetal_brain_s_origin0_unitSpacing"

    accuracy = 6 # decimal places for accuracy of unit tests


    """
    Playground
    """
    angle_x = 0
    # angle_x = np.pi/20
    angle_y = 0
    # angle_y = np.pi/20
    # angle_z = 0
    angle_z = np.pi/10

    # angle = np.pi/20
    # translation = (1,2)
    # translation = (0,0)
    # center = (30,40)
    # center = (0,0)
    # translation_3D = (10,0,0)
    translation_3D = (0,0,0)
    center_3D = (0,0,0)
    # center_3D = (10,30,-10)

    # print("Chosen parameters to get moving image:")
    # print("  translation =  (%r,%r) " %(translation))
    # print("  angle       =  %r deg" %(angle*180/np.pi))
    # print("  center      =  (%r,%r) " %(center))


    ## Load image
    stack = sitk.ReadImage(dir_input + filename_3D + ".nii.gz", sitk.sitkFloat64)

    slice_number = 5
    # fixed = fixed[:,:,slice_number]

    fixed_staple = stack
    fixed_staple = stack[:,:,slice_number:slice_number+2]

    ## Generate test transformation:
    # rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)
    rigid_transform_3D = sitk.Euler3DTransform(center_3D, angle_x, angle_y, angle_z, translation_3D)

    sitkh.print_rigid_transformation(rigid_transform_3D)

    ## Get rigidly transformed image
    fixed_staple_motion = sitkh.get_transformed_image(fixed_staple, rigid_transform_3D)

    
    """
    Registration algorithms
    """
    ## Rigid Registration SimpleITK:
    # final_transform_2D = get_sitk_rigid_registration_transform_2D(fixed, stack)

    ## Rigid Registration NiftyReg:
    # final_transform_2D = get_NiftyReg_rigid_registration_transform_2D(fixed, stack)
    # final_transform_3D = get_NiftyReg_rigid_registration_transform_3D(fixed_staple_motion, stack)

    ## Rigid Registration FLIRT:
    # final_transform_2D = get_FLIRT_rigid_registration_transform_2D(fixed, stack)
    final_transform_3D = get_FLIRT_rigid_registration_transform_3D(fixed_staple_motion, stack)

    fixed_staple_corrected = sitkh.get_transformed_image(fixed_staple_motion, final_transform_3D)

    """
    Resampling
    """
    # warped_registration = sitk.Resample(stack, fixed, final_transform_2D, sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())
    # fixed_staple_warped = sitk.Resample(fixed_staple_motion, stack, final_transform_3D, sitk.sitkNearestNeighbor, 0.0, stack.GetPixelIDValue())

    fixed_staple_warped = sitk.Resample(
        fixed_staple_corrected, 
        stack, 
        sitk.Euler3DTransform(), 
        sitk.sitkNearestNeighbor, 
        0.0, 
        stack.GetPixelIDValue())

    """
    Plot
    """

    # sitkh.show_sitk_image(image_sitk=fixed_staple_warped, overlay_sitk=stack)
    ## Optional: Plot
    # moving_resampled = sitk.Resample(stack, fixed, sitk.Euler2DTransform(), sitk.sitkBSpline, 0.0, fixed.GetPixelIDValue())
    # sitkh.plot_compare_sitk_2D_images(fixed, moving_resampled,1,1)

    ## Optional: Plot outcome
    # sitkh.plot_compare_sitk_2D_images(fixed, warped_registration,2)


    """
    3D registration problems
    """


    """
    Unit tests:
    """
    print("\nUnit tests:\n--------------")
    # unittest.main()
