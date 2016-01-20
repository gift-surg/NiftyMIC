import itk
import SimpleITK as sitk
import numpy as np

import sys
sys.path.append("../src")


dir_input = "data/"
dir_output = "results/"
filename = "fetal_brain_a"
# filename = "CTL_0_baseline_deleted_0.5"

# Define types of input and output pixels and state dimension of images
input_pixel_type = itk.F
output_pixel_type = input_pixel_type

input_dimension = 3
output_dimension = input_dimension

# Define type of input and output image
input_image_type = itk.Image[input_pixel_type, input_dimension]
output_image_type = itk.Image[output_pixel_type, output_dimension]

# Instantiate types of reader and writer
reader_type = itk.ImageFileReader[input_image_type]
writer_type = itk.ImageFileWriter[output_image_type]
image_IO_type = itk.NiftiImageIO

# Create reader and writer
reader = reader_type.New()
writer = writer_type.New()

# Set image IO type to nifti
image_IO = image_IO_type.New()
reader.SetImageIO(image_IO)

# Read image
reader.SetFileName(dir_input + filename + ".nii.gz")
reader.Update()

# Get image
image_itk = reader.GetOutput()

# ITK to NumPy
itk2np = itk.PyBuffer[input_image_type]
nda = itk2np.GetArrayFromImage(image_itk)



# Gaussian Filtering
sigma = 3
filter_type_YVV = itk.SmoothingRecursiveYvvGaussianImageFilter[input_image_type, output_image_type]
filter_YVV = filter_type_YVV.New()
filter_YVV.SetInput(image_itk)
filter_YVV.SetSigma(sigma)
print("OK 1")
filter_YVV.Update()     #Execution of the Recursive Gaussian YVV Filter
print("OK 2")