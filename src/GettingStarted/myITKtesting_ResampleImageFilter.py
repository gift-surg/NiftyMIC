import itk
import numpy as np

dir_input = "data/"
dir_output = "results/"
filename = "BrainWeb_2D"

# Define types of input and output pixels and state dimension of images
input_pixel_type = itk.UC
output_pixel_type = itk.UC

input_dimension = 2
output_dimension = input_dimension

# Define type of input and output image
input_image_type = itk.Image[input_pixel_type, input_dimension]
output_image_type = itk.Image[output_pixel_type, output_dimension]

# Instantiate types of reader and writer
reader_type = itk.ImageFileReader[input_image_type]
writer_type = itk.ImageFileWriter[output_image_type]

# Create reader and writer
reader = reader_type.New()
writer = writer_type.New()

# Set file names to be read and written
reader.SetFileName(dir_input + filename + ".png")
writer.SetFileName(dir_output + filename + "_ResampleImageFilter.png")

# Store input image
input_image = reader.GetOutput()
# input_image.SetPixel((20,5),0)    # leads to crash!?

# Resample Image Filter
filter_type = itk.ResampleImageFilter[input_image_type, output_image_type]
filter = filter_type.New()
filter.SetInput(reader.GetOutput())


# transform_type = itk.AffineTransform.D2
# transform = transform_type.New()
# translation = (-30,-50)               #translation in x and y in millimeters
# transform.Translate(translation)
# filter.SetTransform(transform)

interpolator_type = itk.NearestNeighborInterpolateImageFunction.IUC2D #Input image type: UC 2D
interpolator = interpolator_type.New()
filter.SetInterpolator(interpolator)

filter.SetDefaultPixelValue(100)  #set values for pixels outside the image after trafo

spacing = (1,1)
origin = (0,0)
filter.SetOutputSpacing(spacing)
filter.SetOutputOrigin(origin)
filter.SetOutputDirection(input_image.GetDirection())



size = (300,500)        #number of pixels along x and y
filter.SetSize(size)


filter.Update() #Execution of the filter

# Write Image
writer.SetInput(filter.GetOutput())
writer.Update()