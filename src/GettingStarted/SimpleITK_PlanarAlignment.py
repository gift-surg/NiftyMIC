import SimpleITK as sitk
import nibabel as nib
import numpy as np

import sys
sys.path.append("../v1_20150915/")

from FileAndImageHelpers import *

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


def compute_new_3D_origin_from_2D_alignment(transform_2D, slice_3D):
    # Get parameters of 2D registration
    angle_z, translation_x, translation_y = transform_2D.GetParameters()
    
    # Expand obtained translation to 3D vector
    translation_2D = np.array([translation_x, translation_y, 0])

    # Fetch information of current position in physical space of 3D slice
    origin = slice_3D.GetOrigin()
    spacing = slice_3D.GetSpacing()
    direction_matrix = np.array(slice_3D.GetDirection()).reshape(3,3)

    # Update origin of 3D slice given the planar registration 
    # return direction_matrix.dot(translation_2D*spacing) + origin #automatic cast to np.arrays
    return direction_matrix.dot(-translation_2D) + origin #automatic cast to np.arrays
    # return translation_2D + origin #automatic cast to np.arrays


def compute_new_3D_direction_matrix_from_2D_alignment(transform_2D, slice_3D):
    # Get parameters of 2D registration
    angle_z, translation_x, translation_y = transform_2D.GetParameters()
    center = transform_2D.GetFixedParameters()   #center of 2D rotation 

    # Create 3D rigid transformation
    rigid_transform_3D = sitk.Euler3DTransform()
    rigid_transform_3D.SetRotation(0, 0, angle_z)
    rigid_transform_3D.SetTranslation((translation_x, translation_y, 0))
    rigid_transform_3D.SetCenter((center[0], center[1], 0))

    transform_3D_rotation_matrix = np.array(rigid_transform_3D.GetMatrix()).reshape(3,3)
    # print transform_3D_rotation_matrix


    # Fetch information of current position in physical space of 3D slice
    direction_matrix = np.array(slice_3D.GetDirection()).reshape(3,3)

    # Compute new direction matrix and return in SimpleITK format
    return (direction_matrix.dot(transform_3D_rotation_matrix)).reshape(-1)


def test_planar_alignment(transform_2D, slice_3D):
    Nx,Ny,Nz = slice_3D.GetSize()

    e_0 = (0,0,0)
    e_x = (Nx,0,0)
    e_y = (0,Ny,0)
    # e_z = (0,0,Nz)

    a_0 = np.array(slice_3D.TransformIndexToPhysicalPoint(e_0))
    a_x = np.array(slice_3D.TransformIndexToPhysicalPoint(e_x)) - a_0
    a_y = np.array(slice_3D.TransformIndexToPhysicalPoint(e_y)) - a_0
    # a_z = np.array(slice_3D.TransformIndexToPhysicalPoint(e_z)) - a_0

    origin = compute_new_3D_origin_from_2D_alignment(transform_2D, slice_3D)
    direction = compute_new_3D_direction_matrix_from_2D_alignment(transform_2D, slice_3D)

    slice_3D.SetOrigin(origin)
    slice_3D.SetDirection(direction)

    b_0 = np.array(slice_3D.TransformIndexToPhysicalPoint(e_0))
    b_x = np.array(slice_3D.TransformIndexToPhysicalPoint(e_x)) - b_0
    b_y = np.array(slice_3D.TransformIndexToPhysicalPoint(e_y)) - b_0
    # b_z = np.array(slice_3D.TransformIndexToPhysicalPoint(e_z)) - b_0
    b_ortho = np.cross(b_x,b_y)

    print b_0

    print("e_x dot e_y = 0?: Result = " + str(np.dot(e_x,e_y)))
    print("a_x dot a_y = 0?: Result = " + str(np.dot(a_x,a_y)))
    print("b_x dot b_y = 0?: Result = " + str(np.dot(b_x,b_y)))
    print("b_ortho dot a_x = 0?: Result = " + str(np.dot(b_ortho,a_x)))
    print("b_ortho dot a_y = 0?: Result = " + str(np.dot(b_ortho,a_y)))

    # print("\n")

    angle_z, translation_x, translation_y = transform_2D.GetParameters()
    spacing = slice_3D.GetSpacing()
    print("norm(a_0 - b_0) = " + str(np.linalg.norm(a_0 - b_0)))
    print("norm(t_x,t_y) = " + str(np.linalg.norm((translation_x,translation_y))))


"""
Use SimpleITK to register images in-plane
"""

dir_input = "../../results/input_data/"
dir_output = "results/"
filename =  "0"


stack = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat32)
# stack_aligned_planar = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat32)

stack_mask = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)
# stack_mask_aligned_planar = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)

N = stack.GetSize()[-1]

# stack.SetSpacing((0.5,0.5,0.5))


i = 0
step = 1

slice_3D_fixed = stack[:,:,i:i+1]
slice_3D_moving = stack[:,:,i+step:i+step+1]

# slice_3D_fixed = stack
# slice_3D_moving = stack

slice_2D_fixed = slice_3D_fixed[:,:,0]
slice_2D_moving = slice_3D_moving[:,:,0]


# stack_nda = sitk.GetArrayFromImage(stack_aligned_planar) #now indexed as [z,y,x]!
# stack_mask_nda = sitk.GetArrayFromImage(stack_mask_aligned_planar) #now indexed as [z,y,x]!

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

final_transform = registration_method.Execute(
    sitk.Cast(slice_2D_fixed, sitk.sitkFloat32), sitk.Cast(slice_2D_moving, sitk.sitkFloat32))

warped = sitk.Resample(slice_2D_moving, slice_2D_fixed, final_transform, sitk.sitkLinear, 0.0, slice_2D_moving.GetPixelIDValue())

warped_origin = warped.GetOrigin()
slice_2D_moving_origin = slice_2D_moving.GetOrigin()

print np.linalg.norm(warped_origin-np.array(slice_2D_moving_origin))


"""
Update image information
"""

angle, translation_x, translation_y = final_transform.GetParameters()
translation_2D = np.array((translation_x, translation_y))
print("Translation = " +str(translation_2D))


# print slice_3D_fixed.TransformContinuousIndexToPhysicalPoint((translation_x, translation_y,0))

# angle = np.pi
# translation_x = 50
# translation_y = 25

# final_transform.SetParameters((angle,translation_x, translation_y))

origin_update = compute_new_3D_origin_from_2D_alignment(final_transform, slice_3D_moving)
direction_update = compute_new_3D_direction_matrix_from_2D_alignment(final_transform, slice_3D_moving)

sitk.WriteImage(slice_3D_moving, os.path.join(dir_output, filename+"_"+str(i)+".nii.gz"))


slice_3D_moving.SetOrigin(origin_update)
slice_3D_moving.SetDirection(direction_update)

sitk.WriteImage(slice_3D_moving, os.path.join(dir_output, filename+"_"+str(i)+"_aligned_planar.nii.gz"))

test_planar_alignment(final_transform, slice_3D_fixed)


# warped = sitk.Resample(slice_2D_moving, slice_2D_fixed, final_transform, sitk.sitkLinear, 0.0, slice_2D_moving.GetPixelIDValue())
# warped_mask = sitk.Resample(moving_mask, fixed_mask, final_transform, sitk.sitkNearestNeighbor, 0.0, moving_mask.GetPixelIDValue())

# # for i in xrange(0,N-1):
# for i in xrange(35,36):
#     if i == N-2:
#         step=1

#     fixed = stack_aligned_planar[:,:,i]
#     moving = stack_aligned_planar[:,:,i+step]

#     fixed_mask = stack_mask_aligned_planar[:,:,i]
#     moving_mask = stack_mask_aligned_planar[:,:,i+step]

    

   

#     stack_nda[i+step,:,:] = sitk.GetArrayFromImage(warped)
#     stack_mask_nda[i+step,:,:] = sitk.GetArrayFromImage(warped_mask)

#     # plot_comparison_of_reference_and_warped_image(
#         # fixed=sitk.GetArrayFromImage(fixed), warped=sitk.GetArrayFromImage(warped), fig_id=2)

#     # myshow(fixed-moving)

#     print("Iteration " + str(i+1) + "/" + str(N-1) + ":")
#     print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
#     print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
#     print("\n")

#     fig = plot_comparison_of_reference_and_warped_image(
#         fixed=sitk.GetArrayFromImage(stack)[i,:,:]-sitk.GetArrayFromImage(stack)[i+1,:,:], 
#         warped=sitk.GetArrayFromImage(fixed)-sitk.GetArrayFromImage(warped),
#         fixed_title="without registration between Slice " + str(i) + " and " + str(i+step),
#         warped_title="rigidly registered",
#         fig_id=1)
#     fig.canvas.draw()

#     fig = plot_comparison_of_reference_and_warped_image(
#         fixed=sitk.GetArrayFromImage(stack_mask)[i,:,:]-sitk.GetArrayFromImage(stack_mask)[i+1,:,:], 
#         warped=sitk.GetArrayFromImage(fixed_mask)-sitk.GetArrayFromImage(warped_mask),
#         fixed_title="without registration between Slice " + str(i) + " and " + str(i+step),
#         warped_title="rigidly registered",
#         fig_id=2)
#     fig.canvas.draw()
#     # time.sleep(1)

#     stack_aligned_planar = sitk.GetImageFromArray(stack_nda)
#     stack_aligned_planar.CopyInformation(stack)

#     stack_mask_aligned_planar = sitk.GetImageFromArray(stack_mask_nda)
#     stack_mask_aligned_planar.CopyInformation(stack_mask)

# # imshow(sitk.GetArrayFromImage(fixed)-sitk.GetArrayFromImage(moving),cmap=cm.gray)

# # myshow(fixed-warped)

# sitk.WriteImage(stack_aligned_planar, os.path.join(dir_output,filename+"_aligned_planar.nii.gz"))
# sitk.WriteImage(stack_mask_aligned_planar, os.path.join(dir_output,filename+"_mask_aligned_planar.nii.gz"))

# plt.show()