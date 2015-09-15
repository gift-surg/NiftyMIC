
import itk
# from sys import argv


# pixelType = itk.UC                  #unsigned character
# imageType = itk.Image[pixelType, 2]

# readerType = itk.ImageFileReader[imageType]
# writerType = itk.ImageFileWriter[imageType]
# reader = readerType.New()
# writer = writerType.New()
# reader.SetFileName( argv[1] )
# writer.SetFileName( argv[2] )
# writer.SetInput( reader.GetOutput() )
# writer.Update()

dir_input = "../../results/input_data/"
dir_output = "results/"
filename = "0"

# Define types of input and output pixels and state dimension of images
input_pixel_type = itk.F #F.. float, D...double, UD...long double
output_pixel_type = input_pixel_type

input_dimension = 3
output_dimension = 2

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
image_IO = image_IO_type.New()

# Set file names to be read and written
reader.SetFileName(dir_input + filename + ".nii.gz")
writer.SetFileName(dir_output + filename + "_test.nii.gz")

# Set image IO type to nifti
reader.SetImageIO(image_IO)

reader.Update()

# Extract 2D image from 3D volume
filter_type = itk.ExtractImageFilter[input_image_type, output_image_type]
filter = filter_type.New()
filter.InPlaceOn()
filter.SetDirectionCollapseToSubmatrix()

reader.UpdateOutputInformation()
input_region = reader.GetOutput().GetLargestPossibleRegion() #that's still 3D here

size = input_region.GetSize()
size[2] = 0                         #image reduction in z-direction! => extract 2D image

start = input_region.GetIndex()     #output: start = itkIndex3 ([0, 0, 0])
slice_number = 15                   #specify slice number to be extracted
start[2] = slice_number

# input_image_type.?


# reader.Update()
# stack = reader.GetOutput()

# writer.SetInput(reader.GetOutput())
# writer.Update()




