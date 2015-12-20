## Websites:
# http://kevin-keraudren.blogspot.co.uk/2013/12/irtk-python.html
# http://www.doc.ic.ac.uk/~kpk09/irtk/
# https://github.com/sk1712/IRTK/tree/master/wrapping/cython

# load the module
import irtk
import SimpleITK as sitk


# read an image

dir_in = "data/"
dir_out = "results/"
img_filename = "placenta_s"

img = irtk.imread( dir_in+img_filename+".nii.gz" )

print img[0:2,0:2,0:2] # we crop before printing to save space on the page

# we previously read the image in its type on disk (short == int16)
# we thus had the warning on stderr:
# irtkGenericImage<short>::Read: Ignore slope and intercept, 
# use irtkGenericImage<float> or irtkGenericImage<double> instead
print "Maximum without slope/intercept", float(img.max())

# let's read it again requesting for float ( float == float32 )
img = irtk.imread( dir_in+img_filename+".nii.gz", dtype='float32' )
img_sitk = sitk.ReadImage(dir_in+img_filename+".nii.gz",sitk.sitkFloat32)
print "Maximum with slope/intercept", float(img.max())

# show us a view of the image with some saturation
irtk.imshow( img.saturate(0.01,0.99), filename=dir_out+img_filename+"_quickview.png" )

# now a segmentation
irtk.imshow( img.saturate(0.01,0.99), # input image
            img > 300,               # segmentation labels
            opacity=0.4,
            filename=dir_out+img_filename+"_segmentation.png" )


# Modules from irtk.ext need to be built separately
# by running make in irtk/wrapping/cython/ext

# segmentation colormaps are automatically generated
# the first 10 colors are predefined, the remaining ones are random.
# this works for rview, display and the quick imshow tool
# (the latter works only for ipython notebook or qtconsole, 
# unless you specify a filename for writing to disk)
# let's try a more comlex segmentation like SLIC supervoxels
from irtk.ext.slic import slic
irtk.imshow( img,
             slic( img.gaussianBlurring(2.0), 2000, 10), # segmentation labels
             filename=dir_out+img_filename+"_slic.png" )

# the advantage of a this new interface over the old SWIG wrapper is
# to offer a pythonesque access to IRTK.
# a rule of thumb is that the Python code should be shorter,
# and easier to read than C++

# for instance, if we want to rotate an image around its center (pixel coordinates):

# get the translation
tx,ty,tz = img.ImageToWorld( [(img.shape[2]-1)/2,
                              (img.shape[1]-1)/2,
                              (img.shape[0]-1)/2] )
translation = irtk.RigidTransformation( tx=-tx, ty=-ty, tz=-tz )

# form a rotation
rotation = irtk.RigidTransformation( rx=60 )

# apply the transformations
new_img = img.transform( translation.invert()*rotation*translation,
                         target=img,
                         interpolation='linear' )

irtk.imshow( new_img, filename=dir_out+img_filename+"_rotation.png" )