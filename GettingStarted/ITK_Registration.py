
## import librarties
import itk
import numpy as np

"""
Functions
"""
def get_rigid_registration_transform_3D(fixed_itk, moving_itk):
    pixel_type = itk.D
    dimension = 3
    image_type = itk.Image[pixel_type, dimension]

    registration = itk.ImageRegistrationMethod[image_type, image_type].New()

    # initial_transform = itk.CenteredTransformInitializer[fixed_itk, moving_itk, itk.Euler3DTransform.New()].New()
    initial_transform = itk.Euler3DTransform.New()

    interpolator = itk.LinearInterpolateImageFunction[image_type, pixel_type].New()
    
    metric = itk.MeanSquaresImageToImageMetric[image_type, image_type].New()

    optimizer = itk.RegularStepGradientDescentOptimizer.New()
    # optimizer.SetMaximumStepLength( 4.00 )
    # optimizer.SetMinimumStepLength( 0.01 )
    # optimizer.SetNumberOfIterations( 200 )

    registration.SetInitialTransformParameters(initial_transform.GetParameters())
    registration.SetOptimizer(optimizer)
    registration.SetTransform(initial_transform)
    registration.SetInterpolator(interpolator)
    registration.SetMetric(metric)
    registration.SetMovingImage(moving_itk)
    registration.SetFixedImage(fixed_itk)

    ## Execute registration
    registration.Update()

    ## Get registration transform
    rigid_registration_3D = registration.GetOutput().Get()

    return rigid_registration_3D


def get_resampled_image(fixed_itk, moving_itk, transformation):
    resampler = itk.ResampleImageFilter.New()

    resampler.SetInput(moving_itk)
    resampler.SetTransform(transformation)

    resampler.SetSize(fixed_itk.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_itk.GetOrigin())
    resampler.SetOutputSpacing(fixed_itk.GetSpacing())
    resampler.SetOutputDirection(fixed_itk.GetDirection())
    resampler.SetDefaultPixelValue(0.0)

    warped_itk = resampler.GetOutput()

    return warped_itk


"""
Main
"""
## define input data
dir_input = "data/"
dir_output = "results/"
filename = "fetal_brain_a"

## Define types of input and output pixels and state dimension of images
input_pixel_type = itk.D
output_pixel_type = input_pixel_type

input_dimension = 3
output_dimension = input_dimension

## Define type of input and output image
input_image_type = itk.Image[input_pixel_type, input_dimension]
output_image_type = itk.Image[output_pixel_type, output_dimension]

## Instantiate types of reader and writer
reader_type = itk.ImageFileReader[input_image_type]
writer_type = itk.ImageFileWriter[output_image_type]
image_IO_type = itk.NiftiImageIO

## Create reader and writer
reader_fixed = reader_type.New()
reader_moving = reader_type.New()
writer = writer_type.New()

## Set image IO type to nifti
image_IO = image_IO_type.New()
reader_fixed.SetImageIO(image_IO)
reader_moving.SetImageIO(image_IO)

## Read images
reader_fixed.SetFileName(dir_input + filename + ".nii.gz")
reader_fixed.Update()
fixed_itk = reader_fixed.GetOutput()

reader_moving.SetFileName(dir_input + filename + "_rotated_angle_z.nii.gz")
reader_moving.Update()
moving_itk = reader_moving.GetOutput()

## Register images
rigid_transform_3D = get_rigid_registration_transform_3D(fixed_itk, moving_itk)

## Resample image
warped_itk = get_resampled_image(fixed_itk, moving_itk, rigid_transform_3D)

# trafo = itk.Euler3DTransform.New()
# writer_transformation = itk.TransformFileWriterTemplate[itk.D].New()


## Write warped image
writer.SetFileName(dir_output + filename + "_test.nii.gz")
writer.SetInput(warped_itk)
writer.Update()

