import itk
import SimpleITK as sitk
import time                     

import sys
sys.path.append("../src")

import SimpleITKHelper as sitkh

dir_input = "data/"
dir_output = "results/"
filename = "fetal_brain_a"

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

# Gaussian Filtering
sigma = 3

# Recursive Gaussian YVV Filter
t0 = time.clock()
filter_type_YVV = itk.SmoothingRecursiveYvvGaussianImageFilter[input_image_type, output_image_type]
filter_YVV = filter_type_YVV.New()
filter_YVV.SetInput(image_itk)
filter_YVV.SetSigma(sigma)
print("OK 1")
filter_YVV.Update()     #Execution of the Recursive Gaussian YVV Filter
print("OK 2")
time_elapsed_itk_YVV = time.clock() - t0

# Recursive Gaussian Filter (Deriche)
t0 = time.clock()
filter_type_Deriche = itk.SmoothingRecursiveGaussianImageFilter[input_image_type, output_image_type]
filter_Deriche = filter_type_Deriche.New()
filter_Deriche.SetInput(image_itk)
filter_Deriche.SetSigma(sigma)
filter_Deriche.Update() #Execution of the Recursive Gaussian Filter
time_elapsed_itk_Deriche = time.clock() - t0

print("sdf")

# SimpleITK: Recursive Gaussian Filter (Deriche)
t0 = time.clock()
image_sitk = sitk.ReadImage(dir_input + filename + ".nii.gz")
filter_Deriche_sitk = sitk.SmoothingRecursiveGaussianImageFilter()
filter_Deriche_sitk.SetSigma(sigma)
image_Deriche_sitk = filter_Deriche_sitk.Execute(image_sitk)
time_elapsed_sitk_Deriche = time.clock() - t0

print("Elapsed time for itk filter YVV: %s seconds" %(time_elapsed_itk_YVV))
print("Elapsed time for itk filter Deriche: %s seconds" %(time_elapsed_itk_Deriche))
print("Elapsed time for sitk filter Deriche: %s seconds" %(time_elapsed_sitk_Deriche))

# Write Images
writer.SetFileName(dir_output + filename + "_GaussianFilter_Yvv.nii.gz")
writer.SetInput(filter_YVV.GetOutput())
writer.Update()

writer.SetFileName(dir_output + filename + "_GaussianFilter_Deriche.nii.gz")
writer.SetInput(filter_Deriche.GetOutput())
writer.Update()

sitk.WriteImage(image_Deriche_sitk, dir_output + filename + "_GaussianFilter_Deriche_SimpleITK.nii.gz")

## Compare obtained images
image_Deriche_itk = sitk.ReadImage(dir_output + filename + "_GaussianFilter_Deriche.nii.gz")
image_YVV_itk = sitk.ReadImage(dir_output + filename + "_GaussianFilter_Yvv.nii.gz")

# sitkh.plot_compare_sitk_2D_images(image_Deriche_itk, image_Deriche_sitk)
# sitkh.show_sitk_image(image_Deriche_itk, image_Deriche_sitk)
# sitkh.show_sitk_image(image_Deriche_itk-image_Deriche_sitk)
