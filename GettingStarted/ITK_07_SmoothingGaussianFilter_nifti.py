import itk

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

# Read file
reader.SetFileName(dir_input + filename + ".nii.gz")

# Gaussian Filtering
sigma = 0.8

# Recursive Gaussian YVV Filter
filter_type_YVV = itk.SmoothingRecursiveYvvGaussianImageFilter[input_image_type, output_image_type]
filter_YVV = filter_type_YVV.New()
filter_YVV.SetInput(reader.GetOutput())
filter_YVV.SetSigma(sigma)
filter_YVV.Update()     #Execution of the Recursive Gaussian YVV Filter

# Recursive Gaussian Filter (Deriche)
filter_type_Deriche = itk.SmoothingRecursiveGaussianImageFilter[input_image_type, output_image_type]
filter_Deriche = filter_type_Deriche.New()
filter_Deriche.SetInput(reader.GetOutput())
filter_Deriche.SetSigma(sigma)
filter_Deriche.Update() #Execution of the Recursive Gaussian Filter

# Write Image
writer.SetFileName(dir_output + filename + "_GaussianFilter_Yvv.nii.gz")
writer.SetInput(filter.GetOutput())
writer.Update()

writer.SetFileName(dir_output + filename + "_GaussianFilter_Deriche.nii.gz")
writer.SetInput(filter_Deriche.GetOutput())
writer.Update()
