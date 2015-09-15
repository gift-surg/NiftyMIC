import itk

dir_input = "data/"
dir_output = "results/"
filename = "BrainWeb_2D"

# Define types of input and output pixels and state dimension of images
input_pixel_type = itk.UC
output_pixel_type = itk.UC

input_dimension = 2
output_dimension = 2

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
writer.SetFileName(dir_output + filename + "_BinaryThresholdImageFilter.png")

# Binary Threshold Image Filter
lower_threshold = 100
upper_threshold = 200

outside_value = 0
inside_value = 200

filter_type = itk.BinaryThresholdImageFilter[input_image_type, output_image_type]
filter = filter_type.New()
filter.SetInput(reader.GetOutput())

filter.SetOutsideValue(outside_value)
filter.SetInsideValue(inside_value)
filter.SetLowerThreshold(lower_threshold)
filter.SetUpperThreshold(upper_threshold)

filter.Update() #Execution of the filter

# Write Image
writer.SetInput(filter.GetOutput())
writer.Update()