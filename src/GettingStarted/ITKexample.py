# !/usr/bin/env python
# Example from: http://www.itk.org/Wiki/ITK/Release_4/Wrapping/Examples (11.09.2015)

# run code: python ITKexample.py BrainWeb_2D.png test.bmp
# Doesn't work with ipython!?

# import site
# site.addsitedir('/Users/mebner/development/ITK/ITK-build/Wrapping/Generators/Python/')

"""
# Idea: "To test the wrapping without installation, copy the Wrapping/Generators/Python/WrapITK.pth file to Python's site-packages directory" (http://www.kitware.com/blog/home/post/888)
# But isn't it already installed!?
#
# However, go ahead so as to link itk:
# Source: http://stackoverflow.com/questions/3402168/permanently-add-a-directory-to-pythonpath

# find path directory which Python searches:
SITEDIR=$(python -m site --user-site)

# create if it doesn't exist:
mkdir -p "$SITEDIR"

# copy the pth-file of ITK
cp /Users/mebner/development/ITK/ITK-build/Wrapping/Generators/Python/WrapITK.pth $SITEDIR
"""



import itk
from sys import argv


pixelType = itk.UC                  #unsigned character
imageType = itk.Image[pixelType, 2]

readerType = itk.ImageFileReader[imageType]
writerType = itk.ImageFileWriter[imageType]
reader = readerType.New()
writer = writerType.New()
# reader.SetFileName( argv[1] )
# writer.SetFileName( argv[2] )
writer.SetInput( reader.GetOutput() )
writer.Update()