#!/usr/bin/python

## \file ITK_05_ResampleImageFilter_Rotation.py
#  \brief Figure out how to perform simple rotation of an image with subsequent resampling
#
#  \author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date: 

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
writer.SetFileName(dir_output + filename + "_ResampleImageFilter_Rotation.png")

reader.Update()

# Extract information of input image
input_image = reader.GetOutput()
origin = input_image.GetOrigin()
spacing = input_image.GetSpacing()
direction = input_image.GetDirection()
size = input_image.GetLargestPossibleRegion().GetSize() #voxels in x and y direction

# Resample Image Filter
filter_type = itk.ResampleImageFilter[input_image_type, output_image_type]
filter = filter_type.New()
filter.SetInput(input_image)

filter.SetOutputOrigin(origin)
filter.SetOutputSpacing(spacing)
filter.SetOutputDirection(direction)
filter.SetSize(size)

# Move the origin of the coordinate system (otherwis rotation  around origin of physical coordinates)
transform_type = itk.AffineTransform.D2
transform = transform_type.New()

image_center_x = origin[0] + spacing[0]*size[0]/2.0
image_center_y = origin[1] + spacing[1]*size[1]/2.0
translation_1 = (-image_center_x, -image_center_y)

transform.Translate(translation_1)

# Apply rotation
angle = np.pi/4
transform.Rotate2D(angle, False)       #False: apply rotation after current transform content

# Move origin back to its previous location
translation_2 = (image_center_x, image_center_y)
transform.Translate(translation_2, False)

# 
filter.SetTransform(transform)

# From here: Just additional stuff, to make output image bigger for subsequent translation etc
translation = (30,-50)               #translation in x and y in millimeters
transform.Translate(translation, False)
filter.SetTransform(transform)

interpolator_type = itk.NearestNeighborInterpolateImageFunction.IUC2D #Input image type: UC 2D
interpolator = interpolator_type.New()
filter.SetInterpolator(interpolator)

filter.SetDefaultPixelValue(100)  #set values for pixels outside the image after trafo

spacing = (1,1)
origin = (0,0)
filter.SetOutputSpacing(spacing)
filter.SetOutputOrigin(origin)

size = (300,500)        #number of pixels along x and y
filter.SetSize(size)


filter.Update() #Execution of the filter (I guess not necessary due to subsequent line)

# Write Image
writer.SetInput(filter.GetOutput())
writer.Update()