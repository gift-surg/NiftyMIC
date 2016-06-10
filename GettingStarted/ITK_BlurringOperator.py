#!/usr/bin/python

## \file ITK_BlurringOperator.py
#  \brief Figure out how to apply the blurring operator with different orientations on the HR volume
#
#  \author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date: January 2016

## Import libraries
import SimpleITK as sitk
import itk
# import nibabel as nib
import numpy as np
import unittest
import math                     # for error function (erf)
from scipy.stats import norm    # for normal distribution

import os                       # used to execute terminal commands in python
import sys
import time
sys.path.append("../src/")

## Import modules from src-folder
import SimpleITKHelper as sitkh

## Change viewer for sitk.Show command
#%env SITK_SHOW_COMMAND /Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP

"""
Functions
"""
## Compute the covariance matrix modelling the PSF in-plane and through-plane of a stack
#  The PSF is modelled as Gaussian with
#       FWHM = 1.2*in-plane-resolution (in-plane)
#       FWHM = slice thickness (through-plane)
#  \param[in] stack stack defining the PSF either as itk or sitk image
#  \return Covariance matrix representing the PSF modelled as Gaussian
def get_PSF_covariance_matrix(stack):
    spacing = np.array(stack.GetSpacing())

    ## Compute Gaussian to approximate in-plane PSF:
    sigma_x2 = (1.2*spacing[0])**2/(8*np.log(2))
    sigma_y2 = (1.2*spacing[1])**2/(8*np.log(2))

    if len(spacing) > 2:
        ## Compute Gaussian to approximate through-plane PSF:
        sigma_z2 = spacing[2]**2/(8*np.log(2))

        Cov = np.diag([sigma_x2, sigma_y2, sigma_z2])

    else:
        Cov = np.diag([sigma_x2, sigma_y2])

    return Cov


## Compute rotated covariance matrix which expresses the PSF of the stack
#  in the coordinates of the HR_volume
#  \param[in] HR_volume_sitk current isotropic HR volume
#  \param[in] slice slice which is aimed to be simulated according to the slice acquisition model
#  \param[in] Cov Covariance matrix modelling the Gaussian PSF of an acquired slice
#  \return Covariance matrix U*Sigma_diag*U' where U represents the
#          orthogonal trafo between slice and HR_volume
def get_PSF_rotated_covariance_matrix(HR_volume, slice, Cov):

    ## SimpleITK objects    
    try:
        dim = slice.GetDimension()
        direction_matrix_HR_volume = np.array(HR_volume.GetDirection()).reshape(dim,dim)
        direction_matrix_slice = np.array(slice.GetDirection()).reshape(dim,dim)

    ## ITK objects    
    except Exception as e:
        dim = slice.GetBufferedRegion().GetImageDimension()
        direction_matrix_HR_volume = get_numpy_array_from_itk_matrix(HR_volume.GetDirection())
        direction_matrix_slice = get_numpy_array_from_itk_matrix(slice.GetDirection())

    ## Compute rotation matrix to express the PSF in the coordinate system of the HR volume
    U = direction_matrix_HR_volume.transpose().dot(direction_matrix_slice)

    # print("U = \n%s" % U);

    return U.dot(Cov).dot(U.transpose())


## Compute covariance matrix ready to use for subsequent interpolation step
#  within the resampling process
#  \param[in] HR_volume_sitk current isotropic HR volume
#  \param[in] slice_sitk slice which is aimed to be simulated according to the slice acquisition model
#  \param[in] Cov Covariance matrix modelling the Gaussian PSF of an acquired slice
#  \return Covariance matrix S*U*Sigma_diag*U'*S where U represents the
#          orthogonal trafo between stack and HR_volume and S the scaling matrix
def get_PSF_scaled_inverse_rotated_covariance_matrix(HR_volume_sitk, slice_sitk, Cov):
    spacing = np.array(HR_volume_sitk.GetSpacing())
    S = np.diag(spacing)
    Sigma_inv = np.linalg.inv(get_PSF_rotated_covariance_matrix(HR_volume_sitk, slice_sitk, Cov))

    return S.dot(Sigma_inv).dot(S)



def get_interpolator(interpolator_type, HR_volume_sitk=None, slice_sitk=None):

    ## Nearest neighbour
    if interpolator_type is 'NearestNeighbour':
        interpolator_type = itk.NearestNeighborInterpolateImageFunction.ID3D #Input image type: Float 3D
        interpolator = interpolator_type.New()
    
    ## Linear
    elif interpolator_type is 'Linear':
        interpolator_type = itk.LinearInterpolateImageFunction[input_image_type, input_pixel_type]
        interpolator = interpolator_type.New()

    ## Gaussian
    elif interpolator_type is 'Gaussian':
        alpha = 1
        Sigma_PSF = get_PSF_covariance_matrix(slice_sitk)
        Sigma_rotated = get_PSF_rotated_covariance_matrix(HR_volume_sitk, slice_sitk, Sigma_PSF)
        # Sigma_rotated = Sigma_PSF
        Sigma_diag = np.sqrt(Sigma_rotated.diagonal())

        print("Sigma_PSF = \n%s" %Sigma_PSF)
        print("Sigma_rotated = \n%s" %Sigma_rotated)
        print("Sigma_diag = \n%s" %Sigma_diag)

        ## Define type of input and output image
        dim = HR_volume_sitk.GetDimension()
        pixel_type = itk.D
        image_type = itk.Image[pixel_type, dim]

        interpolator_type = itk.GaussianInterpolateImageFunction[image_type, pixel_type]
        # interpolator_type = itk.GaussianInterpolateImageFunction.ID3D #Input image type: Float 3D
        interpolator = interpolator_type.New()
        interpolator.SetAlpha(alpha)
        interpolator.SetSigma(Sigma_diag)

        # print("alpha = %s" %interpolator.GetAlpha())
        # print("Sigma = %s" %interpolator.GetSigma())

    ## Oriented Gaussian
    elif interpolator_type is 'OrientedGaussian':
        alpha = 1
        Sigma_PSF = get_PSF_covariance_matrix(slice_sitk)
        Sigma_rotated = get_PSF_rotated_covariance_matrix(HR_volume_sitk, slice_sitk, Sigma_PSF)

        Sigma_diag = np.sqrt(Sigma_rotated.diagonal())

        print("Sigma_PSF = \n%s" %Sigma_PSF)
        print("Sigma_rotated = \n%s" %Sigma_rotated)
        print("Sigma_diag = \n%s" %Sigma_diag)

        ## Define type of input and output image
        dim = HR_volume_sitk.GetDimension()
        pixel_type = itk.D
        image_type = itk.Image[pixel_type, dim]

        interpolator_type = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type]
        # interpolator_type = itk.OrientedGaussianInterpolateImageFunction.ID3D #Input image type: Float 3D
        interpolator = interpolator_type.New()
        interpolator.SetAlpha(alpha)
        interpolator.SetCovariance(Sigma_rotated.flatten())

        # print("alpha = %s" %interpolator.GetAlpha())
        # print("Sigma = %s" %interpolator.GetSigma())

    return interpolator

## Compute blurred index (voxel space) with respect to oriented PSF
#  \param[in] index point in voxedl space
#  \param[in] center
#  \param[in] Sigma ouput of get_scaled_variance_covariance_matrix()
#  \return value proportional to PSF blurred point
def compute_PSF_blurred_point(index, center, Sigma):
    print("\nSigma = \n%s" % (Sigma))
    print("\nSigma.dot(index-center) = \n%s" % (Sigma.dot(index-center)))
    print("(index-center)'*Sigma.dot(index-center) = %s" % (np.sum( (index-center)*Sigma.dot(index-center), 0)))
    print("exp(.) = %s" %(np.exp(-0.5* np.sum( (index-center)*Sigma.dot(index-center), 0))))
    return np.exp(-0.5* np.sum( (index-center)*Sigma.dot(index-center), 0))


## Cast numpy array to itk.Matrix
#  \param[in] matrix_np numpy array to cast
#  \return matrix_itk
def get_itk_matrix_from_numpy_array(matrix_np):
    rows = matrix_np.shape[0]
    cols = matrix_np.shape[1]

    matrix_vnl = itk.vnl_matrix_fixed[itk.D, rows, cols]()

    for i in range(0, rows):
        for j in range(0, cols):
            matrix_vnl.set(i, j, matrix_np[i,j])
    
    matrix_itk = itk.Matrix[itk.D, rows, cols](matrix_vnl)

    return matrix_itk

## Cast itk.Matrix to numpy array
#  \param[in] matrix_itk itk.Matrix
#  \return matrix_np
def get_numpy_array_from_itk_matrix(matrix_itk):
    matrix_vnl = matrix_itk.GetVnlMatrix()

    cols = matrix_vnl.cols()
    rows = matrix_vnl.rows()

    if rows > 1:
        matrix_np = np.zeros((rows,cols))

        for i in range(0, rows):
            for j in range(0, cols):
                matrix_np[i,j] = matrix_itk(i,j)

    else:
        matrix_np = np.zeros(cols)

        for j in range(0, cols):
            matrix_np[j] = matrix_itk(j)

    return matrix_np


## Composite two itk.AffineTransformations
#  \param[in] transform_outer
#  \param[in] transfrom_inner
#  \return transform_outer \circ transform_inner
def get_composited_itk_affine_transform(transform_outer, transform_inner):
    dim = transform_outer.GetCenter().GetPointDimension()

    transform_type = itk.AffineTransform[itk.D, dim]
    transform = transform_type.New()

    fixed_parameters = transform_inner.GetFixedParameters()
    parameters = transform_inner.GetParameters()

    transform.SetParameters(parameters)
    transform.SetFixedParameters(fixed_parameters)

    transform.Compose(transform_outer)

    return transform


## Get itk.AffineTransform encoded in the image
#  \param[in] image_itk
#  \return itk.AffineTransform
def get_itk_affine_transform_from_itk_image(image_itk):
    origin_itk = image_itk.GetOrigin()
    direction_itk = image_itk.GetDirection()
    spacing_itk = image_itk.GetSpacing()
    size_itk = image_itk.GetBufferedRegion().GetSize()
    dim = image_itk.GetBufferedRegion().GetImageDimension()

    S = np.diag(spacing_itk)

    ## Get affine matrix from itk image
    matrix_vnl = itk.vnl_matrix_fixed[itk.D, dim, dim]()
    for i in range(0, dim):
        for j in range(0, dim):
            value = direction_itk(i,j)*spacing_itk[j]
            matrix_vnl.set(i, j, value)
            # print("M(%s,%s) = %s" %(i,j,value))

    ## Create AffineTransform
    transform_type = itk.AffineTransform[itk.D, dim]
    transform = transform_type.New()
    matrix_itk = itk.Matrix[itk.D, dim, dim](matrix_vnl)
    transform.SetMatrix( matrix_itk )

    transform.SetTranslation( origin_itk )
    # print("origin_itk = (%s, %s)" %(origin_itk[0], origin_itk[1]))

    return transform


def get_itk_image_direction_matrix_from_itk_affine_transform(affine_transform_itk, image_itk):
    dim = image_itk.GetBufferedRegion().GetImageDimension()
    spacing = np.array(image_itk.GetSpacing())
    S_inv = np.diag(1/spacing)

    A = get_numpy_array_from_itk_matrix(affine_transform_itk.GetMatrix())

    return get_itk_matrix_from_numpy_array( A.dot(S_inv) )


def get_itk_image_origin_from_itk_affine_transform(affine_transform_itk, image_itk):
    """
    see sitkh.get_sitk_image_origin_from_sitk_affine_transform

    Important: Only tested for center=\0! Not clear how it shall be implemented,
            cf. Johnson2015a on page 551 vs page 107!

    Mostly outcome of application of get_composited_sitk_affine_transform and first transform_inner is image. 
    Therefore, center_composited is always zero on tested functions so far
    """
    dim = image_itk.GetBufferedRegion().GetImageDimension()    

    affine_center = np.array(affine_transform_itk.GetCenter())
    affine_translation = np.array(affine_transform_itk.GetTranslation())
    
    R = get_numpy_array_from_itk_matrix(affine_transform_itk.GetMatrix())

    return affine_center + affine_translation 
    # return affine_center + affine_translation - R.dot(affine_center)


def get_transformed_image(image_itk, transform):
    """
    image_itk.New() and image_itk.Clone() do not work properly!!
    """
    dim = image_itk.GetImageDimension()

    image_type = itk.Image[itk.D, dim]

    image_duplicator = itk.ImageDuplicator[image_type].New()
    image_duplicator.SetInputImage(image_itk)
    image_duplicator.Update()
    image = image_duplicator.GetOutput()

    # image.CopyInformation(image_itk)

    affine_transform = get_itk_affine_transform_from_itk_image(image)
    transform = get_composited_itk_affine_transform(transform, affine_transform)

    direction = get_itk_image_direction_matrix_from_itk_affine_transform(transform, image)
    origin = get_itk_image_origin_from_itk_affine_transform(transform, image)

    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


## print itk.AffineTransform
def print_affine_transform(affine_transform, text="affine transformation"):
    dim = affine_transform.GetCenter().GetPointDimension()
    matrix_itk = affine_transform.GetMatrix()
    
    translation = np.array(affine_transform.GetTranslation())
    center = np.array(affine_transform.GetCenter())
    parameters = np.array(affine_transform.GetParameters())

    matrix = np.zeros((dim,dim))
    for i in range(0, dim):
        for j in range(0, dim):
            matrix[i,j] = matrix_itk(i,j)

    if isinstance(affine_transform, itk.Euler2DTransform[itk.D]) or \
        isinstance(affine_transform, itk.Euler3DTransform[itk.D]):
        print(text + ":")
        print("matrix = \n" + str(matrix))
        print("center = " + str(center))
        print("angle_x, angle_y, angle_z =  (%s, %s, %s) deg" 
            % (parameters[0]*180/np.pi, parameters[1]*180/np.pi, parameters[2]*180/np.pi))
        print("translation = " + str(translation))

    else:
        print(text + ":")
        print("matrix = \n" + str(matrix))
        print("center = " + str(center))
        print("translation = " + str(translation))

    return None


## Compute rotation such that center of (2D or 3D) image remains fixed
#  \param[in] image_itk sitk.Image to rotate (2D or 3D)
#  \param[in] angles_in_deg Scalar (2D) or vector (3D) containing rotation angles. 
#             In 3D: angles_in_deg = (angle_x_deg, angle_y_deg, angle_z_deg)
#  \return itk.AffineTransform[itk.D, 2] or itk.AffineTransform[itk.D, 3]
def get_centered_rotation_itk(image_itk, angles_in_deg):
    origin = np.array(image_itk.GetOrigin())
    spacing = np.array(image_itk.GetSpacing())
    size = np.array(image_itk.GetBufferedRegion().GetSize())
    dim = image_itk.GetBufferedRegion().GetImageDimension()

    ## Trafo to bring image back to origin with axis aligned image borders
    affine = itk.AffineTransform[itk.D, dim].New()
    affine_inv = itk.AffineTransform[itk.D, dim].New()

    affine.SetMatrix(image_itk.GetDirection())
    affine.SetTranslation(image_itk.GetOrigin())
    affine.GetInverse(affine_inv)

    ## Compute centering transformations
    translation1 = itk.AffineTransform[itk.D, dim].New()
    translation2 = itk.AffineTransform[itk.D, dim].New()

    image_center = size*spacing/2.0
    translation1.SetTranslation(-image_center)
    translation2.SetTranslation(image_center)

    ## Construct rotation transform for axis aligned image borders
    # transform_rot = itk.AffineTransform[itk.D, dim].New()
    angles = -np.array(angles_in_deg)*np.pi/180

    if dim == 2:
        rotation = itk.Euler2DTransform[itk.D].New()
        rotation.SetRotation(angles)
    else:
        rotation = itk.Euler3DTransform[itk.D].New()
        rotation.SetRotation(angles[0], angles[1], angles[2])    

    ## Construct composite rotation transform
    transform = get_composited_itk_affine_transform(translation1, affine_inv)
    transform = get_composited_itk_affine_transform(rotation, transform)
    transform = get_composited_itk_affine_transform(translation2, transform)
    transform = get_composited_itk_affine_transform(affine, transform)

    return transform

## Compute rotation such that center of (2D or 3D) image remains fixed
#  \param[in] image_sitk sitk.Image to rotate (2D or 3D)
#  \param[in] angles_in_deg Scalar (2D) or vector (3D) containing rotation angles. 
#             In 3D: angles_in_deg = (angle_x_deg, angle_y_deg, angle_z_deg)
#  \return sitk.AffineTransform(2) or sitk.AffineTransform(3)
def get_centered_rotation_sitk(image_sitk, angles_in_deg):
    origin = np.array(image_sitk.GetOrigin())
    spacing = np.array(image_sitk.GetSpacing())
    size = np.array(image_sitk.GetSize())
    dim = image_sitk.GetDimension()

    ## Trafo to bring image back to origin with axis aligned image borders
    affine = sitk.AffineTransform(dim)
    affine.SetMatrix(image_sitk.GetDirection())
    affine.SetTranslation(image_sitk.GetOrigin())
    affine_inv = sitk.AffineTransform(affine.GetInverse())

    ## Compute centering transformations
    translation1 = sitk.AffineTransform(dim)
    translation2 = sitk.AffineTransform(dim)
    
    image_center = spacing*size/2.0
    translation1.SetTranslation(-image_center)
    translation2.SetTranslation(image_center)

    ## Define rotation
    center = np.zeros(dim)
    translation = np.zeros(dim)
    angles = -np.array(angles_in_deg)*np.pi/180

    if dim == 2:
        rotation = sitk.Euler2DTransform(center, angles, translation)
    else:
        rotation = sitk.Euler3DTransform(center, angles[0], angles[1], angles[2], translation)

    ## Construct composite rotation transform
    transform = sitkh.get_composited_sitk_affine_transform(translation1, affine_inv)
    transform = sitkh.get_composited_sitk_affine_transform(rotation, transform)
    transform = sitkh.get_composited_sitk_affine_transform(translation2, transform)
    transform = sitkh.get_composited_sitk_affine_transform(affine, transform)

    return transform


## Show image with ITK-Snap. Image is saved to /tmp/ for that purpose
#  \param[in] image_itk image to show
#  \param[in] segmentation_itk 
#  \param[in] overlay_itk image which shall be overlayed onto image_itk (optional)
#  \param[in] title filename for file written to /tmp/ (optional)
def show_itk_image(image_itk, segmentation_itk=None, overlay_itk=None, title="test"):
    
    dir_output = "/tmp/"
    # cmd = "fslview " + dir_output + title + ".nii.gz & "

    ## Define type of output image
    pixel_type = itk.D
    dim = image_itk.GetBufferedRegion().GetImageDimension()
    image_type = itk.Image[pixel_type, dim]

    ## Define writer
    writer = itk.ImageFileWriter[image_type].New()
    # writer_2D.Update()
    # image_IO_type = itk.NiftiImageIO

    ## Write image_itk
    writer.SetInput(image_itk)
    writer.SetFileName(dir_output + title + ".nii.gz")
    writer.Update()

    if overlay_itk is not None and segmentation_itk is None:
        ## Write overlay_itk:
        writer.SetInput(overlay_itk)
        writer.SetFileName(dir_output + title + "_overlay.nii.gz")
        writer.Update()

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-o " + dir_output + title + "_overlay.nii.gz " \
            "& "
    
    elif overlay_itk is None and segmentation_itk is not None:
        ## Write segmentation_itk:
        writer.SetInput(segmentation_itk)
        writer.SetFileName(dir_output + title + "_segmentation.nii.gz")
        writer.Update()

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-s " + dir_output + title + "_segmentation.nii.gz " \
            + "& "

    elif overlay_itk is not None and segmentation_itk is not None:
        ## Write overlay_itk:
        writer.SetInput(overlay_itk)
        writer.SetFileName(dir_output + title + "_overlay.nii.gz")
        writer.Update()

        ## Write segmentation_itk:
        writer.SetInput(segmentation_itk)
        writer.SetFileName(dir_output + title + "_segmentation.nii.gz")
        writer.Update()

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-s " + dir_output + title + "_segmentation.nii.gz " \
            + "-o " + dir_output + title + "_overlay.nii.gz " \
            + "& "

    else:
        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            "& "

    os.system(cmd)

    return None


## Create an image which only contains one centered dot with max intensity
#  Works for 2D and 3D images
#  \param[in] image_sitk sitk::Image serving which gets cleared out
#  \param[in] filename filename for the generated single-dot image (optional)
#  \return single-dot image is written to the specified location if given
def create_image_with_single_dot(image_sitk, filename=None):
    nda = sitk.GetArrayFromImage(image_sitk)
    nda[:] = 0
    shape = np.array(nda.shape)

    ## This choice of midpoint will center the max-intensity voxel with the
    #  center-point the viewers ITK-Snap and FSLView use.
    midpoint = tuple((shape/2.).astype(int))

    nda[midpoint]=100 #could also be seen as percent!

    image_SingleDot = sitk.GetImageFromArray(nda)
    image_SingleDot.CopyInformation(image_sitk)

    sitkh.show_sitk_image(image_SingleDot, filename_out="SingleDot_" + str(len(shape)) + "D")

    if filename is not None:
        sitk.WriteImage(image_SingleDot, filename)

## Create an image which contains a cross in the center of the image
#  \param[in] image_sitk sitk::Image serving which gets cleared out
#  \param[in] filename filename for the generated single-dot image (optional)
#  \return single-dot image is written to the specified location if given
def create_image_with_cross_2D(image_sitk, filename=None):
    nda = sitk.GetArrayFromImage(image_sitk)
    nda[:] = 0
    shape = np.array(nda.shape)

    ## This choice of midpoint will center the max-intensity voxel with the
    #  center-point the viewers ITK-Snap and FSLView use.
    midpoint = tuple((shape/2.).astype(int))

    x_range = 5
    y_range = 15
    nda[midpoint]=100 #could also be seen as percent!
    nda[
        midpoint[0]-x_range:midpoint[0]+x_range+1,
        midpoint[1]
        ]=100 #could also be seen as percent!
    nda[
        midpoint[0],
        midpoint[1]-y_range:midpoint[1]+y_range+1
        ]=100 #could also be seen as percent!

    image_SingleDot = sitk.GetImageFromArray(nda)
    image_SingleDot.CopyInformation(image_sitk)

    sitkh.show_sitk_image(image_SingleDot, filename_out="Cross_2D")

    if filename is not None:
        sitk.WriteImage(image_SingleDot, filename)


## Create an image which contains a cross in the center of the image
#  \param[in] image_sitk sitk::Image serving which gets cleared out
#  \param[in] filename filename for the generated single-dot image (optional)
#  \return single-dot image is written to the specified location if given
def create_image_with_cross_3D(image_sitk, filename=None):
    nda = sitk.GetArrayFromImage(image_sitk)
    nda[:] = 0
    shape = np.array(nda.shape)

    ## This choice of midpoint will center the max-intensity voxel with the
    #  center-point the viewers ITK-Snap and FSLView use.
    midpoint = tuple((shape/2.).astype(int))

    x_range = 5
    y_range = 15
    z_range = 25
    nda[midpoint]=100 #could also be seen as percent!
    nda[
        midpoint[0]-x_range:midpoint[0]+x_range+1,
        midpoint[1],
        midpoint[2]
        ]=100
    nda[
        midpoint[0],
        midpoint[1]-y_range:midpoint[1]+y_range+1,
        midpoint[2]
        ]=100
    nda[
        midpoint[0],
        midpoint[1],
        midpoint[2]-z_range:midpoint[2]+z_range+1
        ]=100

    image_SingleDot = sitk.GetImageFromArray(nda)
    image_SingleDot.CopyInformation(image_sitk)

    sitkh.show_sitk_image(image_SingleDot, filename_out="Cross_3D")

    if filename is not None:
        sitk.WriteImage(image_SingleDot, filename)

"""
Unit Test Class
"""

class TestUM(unittest.TestCase):

    accuracy = 10
    dir_input = "data/"
    dir_output = "results/"


    def setUp(self):
        pass

    def test_01_get_Sigma_PSF(self):
        filename_stack = "FetalBrain_stack2_registered"

        stack_sitk = sitk.ReadImage(self.dir_input + filename_stack + ".nii.gz", sitk.sitkFloat32)
        spacing = np.array(stack_sitk.GetSpacing())
        
        Sigma = get_PSF_covariance_matrix(stack_sitk)
        FWHM = np.array([1.2*spacing[0], 1.2*spacing[1], spacing[2]])

        for i in range(0, len(Sigma)):
            M = norm.pdf(0, loc=0, scale=np.sqrt(Sigma[i,i]))
            M_half_left = norm.pdf(-FWHM[i]/2., loc=0, scale=np.sqrt(Sigma[i,i]))
            M_half_right = norm.pdf(FWHM[i]/2., loc=0, scale=np.sqrt(Sigma[i,i]))

            ## Check results      
            self.assertEqual(np.around(
                abs( M - (M_half_left + M_half_right) )
                , decimals = self.accuracy), 0 )


    ## Compare results of itkOrientedGaussianInterpolateImageFunction with 
    #  those of itkGaussianInterpolateImageFunction on the image SingleDot_2D_50.nii.gz
    #  for a diagonal covariance matrix
    #  In addition, test whether SetSigma and SetCovariance of itkOriented ...
    #  yield the same value
    def test_02_itkOrientedGaussianInterpolateImageFunction_2D(self):
        # filename = "BrainWeb_2D"
        filename = "SingleDot_2D_50"

        Cov = np.zeros((2,2))
        Cov[0,0] = 10
        Cov[1,1] = 2
        Sigma = np.sqrt(Cov.diagonal())

        alpha = 3 #cutoff-distance

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 2

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[image_type]
        writer_type = itk.ImageFileWriter[image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader.SetImageIO(image_IO)

        ## Read images
        reader.SetFileName(self.dir_input + filename + ".nii.gz")
        reader.Update()

        ## Get image
        stack_itk = reader.GetOutput()

        ## Resample Image Filter and Interpolator
        filter_type = itk.ResampleImageFilter[image_type, image_type]

        filter_Gaussian = filter_type.New()
        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[image_type, pixel_type].New()

        filter_OrientedGaussian = filter_type.New()
        filter_OrientedGaussian2 = filter_type.New()
        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_OrientedGaussian2 = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()

        ## Set interpolator parameter 
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(Sigma)

        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetSigma(Sigma)

        interpolator_OrientedGaussian2.SetAlpha(alpha)
        interpolator_OrientedGaussian2.SetCovariance(Cov.flatten())

        ## Set filter
        filter_Gaussian.SetInput(stack_itk)
        filter_Gaussian.SetOutputParametersFromImage(stack_itk)
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)
        filter_Gaussian.Update()

        filter_OrientedGaussian.SetInput(stack_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(stack_itk)
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)
        filter_OrientedGaussian.Update()

        filter_OrientedGaussian2.SetInput(stack_itk)
        filter_OrientedGaussian2.SetOutputParametersFromImage(stack_itk)
        filter_OrientedGaussian2.SetInterpolator(interpolator_OrientedGaussian2)
        filter_OrientedGaussian2.Update()

        ## Show result
        # show_itk_image(image_itk=filter_OrientedGaussian.GetOutput(), overlay_itk=filter_Gaussian.GetOutput(), title=self.id())

        ## Set writer
        writer.SetInput(filter_Gaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_Gaussian.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_OrientedGaussian_SetSigma.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian2.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_OrientedGaussian_SetCovariance.nii.gz")
        # writer.Update()

        ## Get data arrays
        itk2np = itk.PyBuffer[image_type]
        nda_Gaussian = itk2np.GetArrayFromImage(filter_Gaussian.GetOutput()) 
        nda_OrientedGaussian = itk2np.GetArrayFromImage(filter_OrientedGaussian.GetOutput()) 
        nda_OrientedGaussian2 = itk2np.GetArrayFromImage(filter_OrientedGaussian2.GetOutput()) 

        ## Check whether results of Gaussian align roughly with those of OrientedGaussian      
        diff = nda_Gaussian-nda_OrientedGaussian
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


        ## Check whether results of OrientedGaussian (SetSigma) match those of OrientedGaussian (SetCovariance)
        diff = nda_OrientedGaussian-nda_OrientedGaussian2
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        # print("||Sigma-Cov||=%s" %norm_diff)
        self.assertEqual(np.around(
            norm_diff
            , decimals = self.accuracy), 0 )


    ## Compare results of itkOrientedGaussianInterpolateImageFunction with 
    #  those of itkGaussianInterpolateImageFunction on the image SingleDot_2D_50.nii.gz
    #  for a diagonal covariance matrix with REALISTIC PSF values
    #  In addition, test whether SetSigma and SetCovariance of itkOriented ...
    #  yield the same value
    def test_02_itkOrientedGaussianInterpolateImageFunction_2D_realistic_PSF(self):
        # filename = "BrainWeb_2D"
        filename = "SingleDot_2D_50"

        Cov = np.zeros((2,2))
        Cov[0,0] = 0.26786367 #in-plane sigma2 for realistic slices
        Cov[1,1] = 2.67304559 #through-plane sigma2 for realistic slices
        Sigma = np.sqrt(Cov.diagonal())

        alpha = 3 #cutoff-distance

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 2

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[image_type]
        writer_type = itk.ImageFileWriter[image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader.SetImageIO(image_IO)

        ## Read images
        reader.SetFileName(self.dir_input + filename + ".nii.gz")
        reader.Update()

        ## Get image
        stack_itk = reader.GetOutput()

        ## Resample Image Filter and Interpolator
        filter_type = itk.ResampleImageFilter[image_type, image_type]

        filter_Gaussian = filter_type.New()
        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[image_type, pixel_type].New()

        filter_OrientedGaussian = filter_type.New()
        filter_OrientedGaussian2 = filter_type.New()
        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_OrientedGaussian2 = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()

        ## Set interpolator parameter 
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(Sigma)

        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetSigma(Sigma)

        interpolator_OrientedGaussian2.SetAlpha(alpha)
        interpolator_OrientedGaussian2.SetCovariance(Cov.flatten())

        ## Set filter
        filter_Gaussian.SetInput(stack_itk)
        filter_Gaussian.SetOutputParametersFromImage(stack_itk)
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)
        filter_Gaussian.Update()

        filter_OrientedGaussian.SetInput(stack_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(stack_itk)
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)
        filter_OrientedGaussian.Update()

        filter_OrientedGaussian2.SetInput(stack_itk)
        filter_OrientedGaussian2.SetOutputParametersFromImage(stack_itk)
        filter_OrientedGaussian2.SetInterpolator(interpolator_OrientedGaussian2)
        filter_OrientedGaussian2.Update()

        ## Show result
        # show_itk_image(image_itk=filter_OrientedGaussian.GetOutput(), overlay_itk=filter_Gaussian.GetOutput(), title=self.id())

        ## Set writer
        writer.SetInput(filter_Gaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_Gaussian.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_OrientedGaussian_SetRealisticSigma.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian2.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_OrientedGaussian_SetRealisticCovariance.nii.gz")
        # writer.Update()

        ## Get data arrays
        itk2np = itk.PyBuffer[image_type]
        nda_Gaussian = itk2np.GetArrayFromImage(filter_Gaussian.GetOutput()) 
        nda_OrientedGaussian = itk2np.GetArrayFromImage(filter_OrientedGaussian.GetOutput()) 
        nda_OrientedGaussian2 = itk2np.GetArrayFromImage(filter_OrientedGaussian2.GetOutput()) 

        ## Check whether results of Gaussian align roughly with those of OrientedGaussian      
        diff = nda_Gaussian-nda_OrientedGaussian
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


        ## Check whether results of OrientedGaussian (SetSigma) match those of OrientedGaussian (SetCovariance)
        diff = nda_OrientedGaussian-nda_OrientedGaussian2
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        # print("||Sigma-Cov||=%s" %norm_diff)
        self.assertEqual(np.around(
            norm_diff
            , decimals = self.accuracy), 0 )


    ## Compare results of itkOrientedGaussianInterpolateImageFunction with 
    #  those of itkGaussianInterpolateImageFunction on the image SingleDot_2D_50.nii.gz
    #  for a ROTATED diagonal covariance matrix
    def test_02_itkOrientedGaussianInterpolateImageFunction_2D_rotated_PSF(self):
        # filename = "BrainWeb_2D"
        filename = "SingleDot_2D_50"

        alpha = 3               #cutoff-distance
        theta_in_degree = 45    #rotation of Gaussian kernel

        Cov = np.zeros((2,2))
        # Cov[0,0] = 0.26786367   #in-plane sigma2 for realistic slices
        # Cov[1,1] = 2.67304559   #through-plane sigma2 for realistic slices
        Cov[0,0] = 10
        Cov[1,1] = 2


        ## Rotation in mathematically negative direction, but appear then
        #  mathematically positive in ITK-Snap
        theta_in_rad = theta_in_degree*np.pi/180.0
        R = np.zeros(Cov.shape)
        R[0,0] = np.cos(theta_in_rad)
        R[0,1] = np.sin(theta_in_rad)
        R[1,0] = -np.sin(theta_in_rad)
        R[1,1] = np.cos(theta_in_rad)

        Cov = R.dot(Cov).dot(R.transpose())
        Sigma = np.sqrt(Cov.diagonal())

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 2

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[image_type]
        writer_type = itk.ImageFileWriter[image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader.SetImageIO(image_IO)

        ## Read images
        reader.SetFileName(self.dir_input + filename + ".nii.gz")
        reader.Update()

        ## Get image
        stack_itk = reader.GetOutput()

        ## Resample Image Filter and Interpolator
        filter_type = itk.ResampleImageFilter[image_type, image_type]

        filter_Gaussian = filter_type.New()
        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[image_type, pixel_type].New()

        filter_OrientedGaussian = filter_type.New()
        filter_OrientedGaussian2 = filter_type.New()
        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()

        ## Set interpolator parameter 
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(Sigma)

        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetCovariance(Cov.flatten())

        ## Set filter
        filter_Gaussian.SetInput(stack_itk)
        filter_Gaussian.SetOutputParametersFromImage(stack_itk)
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)
        filter_Gaussian.Update()

        filter_OrientedGaussian.SetInput(stack_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(stack_itk)
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)
        filter_OrientedGaussian.Update()

        ## Show result
        # show_itk_image(image_itk=filter_OrientedGaussian.GetOutput(), overlay_itk=filter_Gaussian.GetOutput(), title=self.id())

        ## Set writer
        writer.SetInput(filter_Gaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_Gaussian_rotated_PSF.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_OrientedGaussian_rotated_PSF.nii.gz")
        # writer.Update()

        ## Get data arrays
        itk2np = itk.PyBuffer[image_type]
        nda_Gaussian = itk2np.GetArrayFromImage(filter_Gaussian.GetOutput()) 
        nda_OrientedGaussian = itk2np.GetArrayFromImage(filter_OrientedGaussian.GetOutput()) 

        ## Check whether results of Gaussian align roughly with those of OrientedGaussian      
        diff = nda_Gaussian-nda_OrientedGaussian
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## Rotate image and compare axis aligned with resampled oriented PSF
    def test_02_itkOrientedGaussianInterpolateImageFunction_2D_rotated_Image(self):
        filename = "SingleDot_2D_50"
        # filename = "Cross_2D_50"

        theta_in_degree = 45    #rotation of image

        Cov = np.zeros((2,2))
        Cov[0,0] = 0.26786367   #in-plane sigma2 for realistic slices
        Cov[1,1] = 2.67304559   #through-plane sigma2 for realistic slices
        # Cov[0,0] = 10
        # Cov[1,1] = 2
        Sigma = np.sqrt(Cov.diagonal())

        alpha = 3               #cutoff-distance

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 2

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[image_type]
        writer_type = itk.ImageFileWriter[image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader.SetImageIO(image_IO)

        ## Read images
        reader.SetFileName(self.dir_input + filename + ".nii.gz")
        reader.Update()

        ## Get image
        image_itk = reader.GetOutput()

        ## Rotate image
        rotation_itk = get_centered_rotation_itk(image_itk, theta_in_degree)
        image_rotated_itk = get_transformed_image(image_itk, rotation_itk)

        ## Obtain PSF orientation
        # Cov =  get_PSF_covariance_matrix(image_itk)
        Cov_oriented = get_PSF_rotated_covariance_matrix(image_itk, image_rotated_itk, Cov)

        # print("Cov = \n%s" %Cov)
        # print("Cov_oriented = \n%s" %Cov_oriented)

        ## Resample Image Filter
        filter_type = itk.ResampleImageFilter[image_type, image_type]
        filter_Gaussian = filter_type.New()
        filter_OrientedGaussian = filter_type.New()

        ## Set input image
        filter_Gaussian.SetInput(image_rotated_itk)
        filter_Gaussian.SetOutputParametersFromImage(image_rotated_itk)
        filter_Gaussian.Update()

        filter_OrientedGaussian.SetInput(image_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(image_rotated_itk)
        filter_OrientedGaussian.Update()

        ## Choose interpolator
        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(np.sqrt(Cov.diagonal()))
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)

        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetCovariance(Cov_oriented.flatten())
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)

        ## Apply Filter
        filter_Gaussian.Update()
        filter_OrientedGaussian.Update()

        ## Show result
        # show_itk_image(image_itk=filter_OrientedGaussian.GetOutput(), overlay_itk=filter_Gaussian.GetOutput(), title=self.id())

        ## Set writer
        writer.SetInput(filter_Gaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_Gaussian_rotated_Image.nii.gz")
        writer.Update()

        writer.SetInput(filter_OrientedGaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_02_SingleDot_2D_OrientedGaussian_rotated_Image.nii.gz")
        writer.Update()

        ## Get data arrays
        itk2np = itk.PyBuffer[image_type]
        nda_Gaussian = itk2np.GetArrayFromImage(filter_Gaussian.GetOutput()) 
        nda_OrientedGaussian = itk2np.GetArrayFromImage(filter_OrientedGaussian.GetOutput()) 

        ## Check whether results of Gaussian align roughly with those of OrientedGaussian      
        diff = nda_Gaussian-nda_OrientedGaussian
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## Compare results of itkOrientedGaussianInterpolateImageFunction with 
    #  those of itkGaussianInterpolateImageFunction on the image SingleDot_3D_50.nii.gz
    #  for a diagonal covariance matrix
    #  In addition, test whether SetSigma and SetCovariance of itkOriented ...
    #  yield the same value
    def test_03_itkOrientedGaussianInterpolateImageFunction_3D(self):
        filename = "SingleDot_3D_50"

        Cov = np.zeros((3,3))
        Cov[0,0] = 10
        Cov[1,1] = 5
        Cov[2,2] = 2
        Sigma = np.sqrt(Cov.diagonal())

        alpha = 3 #cutoff-distance

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 3

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[image_type]
        writer_type = itk.ImageFileWriter[image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader.SetImageIO(image_IO)

        ## Read images
        reader.SetFileName(self.dir_input + filename + ".nii.gz")
        reader.Update()

        ## Get image
        stack_itk = reader.GetOutput()

        ## Resample Image Filter and Interpolator
        filter_type = itk.ResampleImageFilter[image_type, image_type]

        filter_Gaussian = filter_type.New()
        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[image_type, pixel_type].New()

        filter_OrientedGaussian = filter_type.New()
        filter_OrientedGaussian2 = filter_type.New()
        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_OrientedGaussian2 = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()

        ## Set interpolator parameter 
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(Sigma)

        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetSigma(Sigma)

        interpolator_OrientedGaussian2.SetAlpha(alpha)
        interpolator_OrientedGaussian2.SetCovariance(Cov.flatten())

        ## Set filter
        filter_Gaussian.SetInput(stack_itk)
        filter_Gaussian.SetOutputParametersFromImage(stack_itk)
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)
        filter_Gaussian.Update()

        filter_OrientedGaussian.SetInput(stack_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(stack_itk)
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)
        filter_OrientedGaussian.Update()

        filter_OrientedGaussian2.SetInput(stack_itk)
        filter_OrientedGaussian2.SetOutputParametersFromImage(stack_itk)
        filter_OrientedGaussian2.SetInterpolator(interpolator_OrientedGaussian2)
        filter_OrientedGaussian2.Update()

        ## Show result
        # show_itk_image(image_itk=filter_OrientedGaussian.GetOutput(), overlay_itk=filter_Gaussian.GetOutput(), title=self.id())

        ## Set writer
        writer.SetInput(filter_Gaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_Gaussian.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_OrientedGaussian_SetSigma.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian2.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_OrientedGaussian_SetCovariance.nii.gz")
        # writer.Update()

        ## Get data arrays
        itk2np = itk.PyBuffer[image_type]
        nda_Gaussian = itk2np.GetArrayFromImage(filter_Gaussian.GetOutput()) 
        nda_OrientedGaussian = itk2np.GetArrayFromImage(filter_OrientedGaussian.GetOutput()) 
        nda_OrientedGaussian2 = itk2np.GetArrayFromImage(filter_OrientedGaussian2.GetOutput()) 

        ## Check whether results of Gaussian align roughly with those of OrientedGaussian      
        diff = nda_Gaussian-nda_OrientedGaussian
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


        ## Check whether results of OrientedGaussian (SetSigma) match those of OrientedGaussian (SetCovariance)
        diff = nda_OrientedGaussian-nda_OrientedGaussian2
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        # print("||Sigma-Cov||=%s" %norm_diff)
        self.assertEqual(np.around(
            norm_diff
            , decimals = self.accuracy), 0 )


    ## Compare results of itkOrientedGaussianInterpolateImageFunction with 
    #  those of itkGaussianInterpolateImageFunction on the image SingleDot_2D_50.nii.gz
    #  for a diagonal covariance matrix with REALISTIC PSF values
    #  In addition, test whether SetSigma and SetCovariance of itkOriented ...
    #  yield the same value
    def test_03_itkOrientedGaussianInterpolateImageFunction_3D_realistic_PSF(self):
        filename = "SingleDot_3D_50"

        Cov = np.zeros((3,3))
        Cov[0,0] = 0.26786367 #in-plane sigma2 for realistic slices
        Cov[1,1] = 0.26786367 #in-plane sigma2 for realistic slices
        Cov[2,2] = 2.67304559 #through-plane sigma2 for realistic slices
        Sigma = np.sqrt(Cov.diagonal())

        alpha = 3 #cutoff-distance

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 3

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[image_type]
        writer_type = itk.ImageFileWriter[image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader.SetImageIO(image_IO)

        ## Read images
        reader.SetFileName(self.dir_input + filename + ".nii.gz")
        reader.Update()

        ## Get image
        stack_itk = reader.GetOutput()

        ## Resample Image Filter and Interpolator
        filter_type = itk.ResampleImageFilter[image_type, image_type]

        filter_Gaussian = filter_type.New()
        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[image_type, pixel_type].New()

        filter_OrientedGaussian = filter_type.New()
        filter_OrientedGaussian2 = filter_type.New()
        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_OrientedGaussian2 = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()

        ## Set interpolator parameter 
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(Sigma)

        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetSigma(Sigma)

        interpolator_OrientedGaussian2.SetAlpha(alpha)
        interpolator_OrientedGaussian2.SetCovariance(Cov.flatten())

        ## Set filter
        filter_Gaussian.SetInput(stack_itk)
        filter_Gaussian.SetOutputParametersFromImage(stack_itk)
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)
        filter_Gaussian.Update()

        filter_OrientedGaussian.SetInput(stack_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(stack_itk)
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)
        filter_OrientedGaussian.Update()

        filter_OrientedGaussian2.SetInput(stack_itk)
        filter_OrientedGaussian2.SetOutputParametersFromImage(stack_itk)
        filter_OrientedGaussian2.SetInterpolator(interpolator_OrientedGaussian2)
        filter_OrientedGaussian2.Update()

        ## Show result
        # show_itk_image(image_itk=filter_OrientedGaussian.GetOutput(), overlay_itk=filter_Gaussian.GetOutput(), title=self.id())

        ## Set writer
        writer.SetInput(filter_Gaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_Gaussian.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_OrientedGaussian_SetRealisticSigma.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian2.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_OrientedGaussian_SetRealisticCovariance.nii.gz")
        # writer.Update()

        ## Get data arrays
        itk2np = itk.PyBuffer[image_type]
        nda_Gaussian = itk2np.GetArrayFromImage(filter_Gaussian.GetOutput()) 
        nda_OrientedGaussian = itk2np.GetArrayFromImage(filter_OrientedGaussian.GetOutput()) 
        nda_OrientedGaussian2 = itk2np.GetArrayFromImage(filter_OrientedGaussian2.GetOutput()) 

        ## Check whether results of Gaussian align roughly with those of OrientedGaussian      
        diff = nda_Gaussian-nda_OrientedGaussian
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


        ## Check whether results of OrientedGaussian (SetSigma) match those of OrientedGaussian (SetCovariance)
        diff = nda_OrientedGaussian-nda_OrientedGaussian2
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        # print("||Sigma-Cov||=%s" %norm_diff)
        self.assertEqual(np.around(
            norm_diff
            , decimals = self.accuracy), 0 )


    ## Rotate image and compare axis aligned with resampled oriented PSF
    def test_03_itkOrientedGaussianInterpolateImageFunction_3D_rotated_Image(self):
        filename = "SingleDot_3D_50"
        # filename = "Cross_3D_50"

        angles_in_deg = (45,-10,-30)    #rotation of image

        Cov = np.zeros((3,3))
        Cov[0,0] = 0.26786367   #in-plane sigma2 for realistic slices
        Cov[1,1] = 0.26786367   #in-plane sigma2 for realistic slices
        Cov[2,2] = 2.67304559   #through-plane sigma2 for realistic slices
        # Cov[0,0] = 10
        # Cov[1,1] = 5
        # Cov[2,2] = 2
        Sigma = np.sqrt(Cov.diagonal())

        alpha = 3               #cutoff-distance

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 3

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[image_type]
        writer_type = itk.ImageFileWriter[image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader.SetImageIO(image_IO)

        ## Read images
        reader.SetFileName(self.dir_input + filename + ".nii.gz")
        reader.Update()

        ## Get image
        image_itk = reader.GetOutput()

        ## Rotate image
        rotation_itk = get_centered_rotation_itk(image_itk, angles_in_deg)
        image_rotated_itk = get_transformed_image(image_itk, rotation_itk)

        ## Obtain PSF orientation
        # Cov =  get_PSF_covariance_matrix(image_itk)
        Cov_oriented = get_PSF_rotated_covariance_matrix(image_itk, image_rotated_itk, Cov)

        # print("Cov = \n%s" %Cov)
        # print("Cov_oriented = \n%s" %Cov_oriented)

        ## Resample Image Filter
        filter_type = itk.ResampleImageFilter[image_type, image_type]
        filter_Gaussian = filter_type.New()
        filter_OrientedGaussian = filter_type.New()

        ## Set input image
        filter_Gaussian.SetInput(image_rotated_itk)
        filter_Gaussian.SetOutputParametersFromImage(image_rotated_itk)
        filter_Gaussian.Update()

        filter_OrientedGaussian.SetInput(image_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(image_rotated_itk)
        filter_OrientedGaussian.Update()

        ## Choose interpolator
        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(np.sqrt(Cov.diagonal()))
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)

        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetCovariance(Cov_oriented.flatten())
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)

        ## Apply Filter
        filter_Gaussian.Update()
        filter_OrientedGaussian.Update()

        ## Show result
        # show_itk_image(image_itk=filter_OrientedGaussian.GetOutput(), overlay_itk=filter_Gaussian.GetOutput(), title=self.id())

        ## Set writer
        writer.SetInput(filter_Gaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_Gaussian_rotated_Image.nii.gz")
        # writer.Update()

        writer.SetInput(filter_OrientedGaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_OrientedGaussian_rotated_Image.nii.gz")
        # writer.Update()

        ## Get data arrays
        itk2np = itk.PyBuffer[image_type]
        nda_Gaussian = itk2np.GetArrayFromImage(filter_Gaussian.GetOutput()) 
        nda_OrientedGaussian = itk2np.GetArrayFromImage(filter_OrientedGaussian.GetOutput()) 

        ## Check whether results of Gaussian align roughly with those of OrientedGaussian      
        diff = nda_Gaussian-nda_OrientedGaussian
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## Rotate image and compare axis aligned with resampled oriented PSF
    def test_03_itkOrientedGaussianInterpolateImageFunction_3D_realistic_HR_vol_slice_scenario(self):
        # filename_stack = "FetalBrain_stack0_registered"
        filename_HR_volume = "FetalBrain_reconstruction_4stacks"
        filename_slice = "FetalBrain_stack0_registered_midslice"

        alpha = 3               #cutoff-distance

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 3

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[image_type]
        writer_type = itk.ImageFileWriter[image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        # reader_stack = reader_type.New()
        reader_HR_volume = reader_type.New()
        reader_slice = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        # reader_stack.SetImageIO(image_IO)
        image_IO = image_IO_type.New()
        reader_HR_volume.SetImageIO(image_IO)
        reader_slice.SetImageIO(image_IO)

        ## Read images
        # reader_stack.Update()
        # reader_stack.SetFileName(self.dir_output + filename_stack + ".nii.gz")

        reader_HR_volume.SetFileName(self.dir_input + filename_HR_volume + ".nii.gz")
        reader_HR_volume.Update()
        
        reader_slice.SetFileName(self.dir_input + filename_slice + ".nii.gz")
        reader_slice.Update()

        ## Get image
        # stack_itk = reader_stack.GetOutput()
        HR_volume_itk = reader_HR_volume.GetOutput()
        slice_itk = reader_slice.GetOutput()

        ## Create single dot in slice
        itk2np = itk.PyBuffer[image_type]
        nda = itk2np.GetArrayFromImage(slice_itk) 
        nda[:] = 0
        midpoint = tuple((np.array(nda.shape)/2.).astype(int))
        nda[midpoint] = 100
        slice_SingleDot_itk = itk2np.GetImageFromArray(nda)
        slice_SingleDot_itk.CopyInformation(slice_itk)

        ## Resample single dot to HR volume space
        filter_type = itk.ResampleImageFilter[image_type, image_type]
        filter = filter_type.New()
        filter.SetInput(slice_SingleDot_itk)
        filter.SetOutputParametersFromImage(HR_volume_itk)
        filter.SetInterpolator(itk.LinearInterpolateImageFunction[image_type, pixel_type].New())
        # filter.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[image_type, pixel_type].New())
        filter.Update()
        HR_volume_SingleDot_itk = filter.GetOutput()

        ## Obtain PSF orientation
        Cov =  get_PSF_covariance_matrix(slice_SingleDot_itk)
        Cov_oriented = get_PSF_rotated_covariance_matrix(HR_volume_SingleDot_itk, slice_SingleDot_itk, Cov)

        print("Cov = \n%s" %Cov)
        print("Cov_oriented = \n%s" %Cov_oriented)

        ## Oriented Gaussian (of practical use)
        t0 = time.clock()
        filter_OrientedGaussian = filter_type.New()
        filter_OrientedGaussian.SetInput(HR_volume_SingleDot_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(slice_SingleDot_itk)
        filter_OrientedGaussian.Update()

        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetCovariance(Cov_oriented.flatten())
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)
        filter_OrientedGaussian.Update()

        time_elapsed = time.clock() - t0
        print("Elapsed time = %s seconds" %(time_elapsed))

        ## Gaussian (for rough comparison)
        filter_Gaussian = filter_type.New()
        filter_Gaussian.SetInput(slice_SingleDot_itk)
        filter_Gaussian.SetOutputParametersFromImage(slice_SingleDot_itk)
        filter_Gaussian.Update()

        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[image_type, pixel_type].New()
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(np.sqrt(Cov.diagonal()))
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)
        filter_Gaussian.Update()

        ## Show result
        show_itk_image(image_itk=filter_OrientedGaussian.GetOutput(), overlay_itk=filter_Gaussian.GetOutput(), title=self.id())

        ## Set writer
        writer.SetInput(filter_Gaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_Gaussian_rotated_Image.nii.gz")
        writer.Update()

        writer.SetInput(filter_OrientedGaussian.GetOutput())
        writer.SetFileName(self.dir_output + "test_03_SingleDot_3D_OrientedGaussian_rotated_Image.nii.gz")
        writer.Update()

        ## Get data arrays
        itk2np = itk.PyBuffer[image_type]
        nda_Gaussian = itk2np.GetArrayFromImage(filter_Gaussian.GetOutput()) 
        nda_OrientedGaussian = itk2np.GetArrayFromImage(filter_OrientedGaussian.GetOutput()) 

        ## Check whether results of Gaussian align roughly with those of OrientedGaussian      
        diff = nda_Gaussian-nda_OrientedGaussian
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)

        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## Flip Cross_2D_50.nii.gz by 180 and resample image to check alignment
    #  with original one given the computations of
    #  get_centered_rotation_itk and get_centered_rotation_sitk.
    def test_04_get_centered_rotation_2D(self):
        filename_2D = "Cross_2D_50"

        angle_in_deg = 180

        ## Define types of input and output pixels and state dimension of images
        input_pixel_type = itk.D
        output_pixel_type = input_pixel_type

        input_dimension = 2
        output_dimension = input_dimension

        ## Define type of input and output image
        input_image_type = itk.Image[input_pixel_type, input_dimension]
        output_image_type = itk.Image[output_pixel_type, output_dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[input_image_type]
        writer_type = itk.ImageFileWriter[output_image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader_stack = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader_stack.SetImageIO(image_IO)

        ## Read images
        reader_stack.SetFileName(self.dir_input + filename_2D + ".nii.gz")
        reader_stack.Update()

        ## Get image
        image_itk = reader_stack.GetOutput()
        image_sitk = sitk.ReadImage(self.dir_input + filename_2D + ".nii.gz")

        ## Compute rotation
        rotation_itk = get_centered_rotation_itk(image_itk, angle_in_deg)
        rotation_sitk = get_centered_rotation_sitk(image_sitk, angle_in_deg)

        ## Rotate image
        image_rotated_sitk = sitkh.get_transformed_image(image_sitk, rotation_sitk)
        image_rotated_itk = get_transformed_image(image_itk, rotation_itk)

        sitk.WriteImage(image_sitk,self.dir_output + "test_04_" + filename_2D + ".nii.gz")
        sitk.WriteImage(image_rotated_sitk,self.dir_output + "test_04_" + filename_2D + "_rotated.nii.gz")

        image_rotated_resampled_sitk = sitk.Resample(image_rotated_sitk, image_sitk,
                sitk.Euler2DTransform(), 
                sitk.sitkNearestNeighbor, 
                0.0,
                image_sitk.GetPixelIDValue())

        ## Show image
        # sitkh.show_sitk_image(image_sitk, None, image_rotated_resampled_sitk)

        ## Test correspondences
        nda = sitk.GetArrayFromImage(image_sitk)
        nda_rotated = sitk.GetArrayFromImage(image_rotated_resampled_sitk)

        self.assertEqual(np.around(
            np.linalg.norm(nda - nda_rotated)
            , decimals = self.accuracy), 0 )


    ## Compares the values between SimpleITK and ITK computation, i.e.
    #  between get_centered_rotation_itk and get_centered_rotation_sitk
    def test_04_get_centered_rotation_2D_compare_SITK_ITK(self):
        filename_2D = "BrainWeb_2D"

        angle_in_deg = 40

        ## Define types of input and output pixels and state dimension of images
        input_pixel_type = itk.D
        output_pixel_type = input_pixel_type

        input_dimension = 2
        output_dimension = input_dimension

        ## Define type of input and output image
        input_image_type = itk.Image[input_pixel_type, input_dimension]
        output_image_type = itk.Image[output_pixel_type, output_dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[input_image_type]
        writer_type = itk.ImageFileWriter[output_image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader_stack = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader_stack.SetImageIO(image_IO)

        ## Read images
        reader_stack.SetFileName(self.dir_input + filename_2D + ".nii.gz")
        reader_stack.Update()

        ## Get image
        stack_itk = reader_stack.GetOutput()
        stack_sitk = sitk.ReadImage(dir_input + filename_2D + ".nii.gz")

        ## Compute rotation
        rotation_itk = get_centered_rotation_itk(stack_itk, angle_in_deg)
        rotation_sitk = get_centered_rotation_sitk(stack_sitk, angle_in_deg)

        ## Fetch parameters
        parameters_itk = np.array(rotation_itk.GetParameters())
        fixed_parameters_itk = np.array(rotation_itk.GetFixedParameters())

        parameters_sitk = np.array(rotation_sitk.GetParameters())
        fixed_parameters_sitk = np.array(rotation_sitk.GetFixedParameters())

        ## Test correspondences
        self.assertEqual(np.around(
            np.linalg.norm(parameters_sitk - parameters_itk)
            , decimals = self.accuracy), 0 )

        self.assertEqual(np.around(
            np.linalg.norm(fixed_parameters_sitk - fixed_parameters_itk)
            , decimals = self.accuracy), 0 )


    ## Compares the values between SimpleITK and ITK computation
    def test_04_get_centered_rotation_3D_compare_SITK_ITK(self):
        # filename = "FetalBrain_reconstruction_4stacks"
        filename = "fetal_brain_a"

        angle_x_in_deg = 10
        angle_y_in_deg = 90
        angle_z_in_deg = 30

        angles_in_deg = (angle_x_in_deg, angle_y_in_deg, angle_z_in_deg)

        ## Define types of input and output pixels and state dimension of images
        input_pixel_type = itk.D
        output_pixel_type = input_pixel_type

        input_dimension = 3
        output_dimension = input_dimension

        ## Define type of input and output image
        input_image_type = itk.Image[input_pixel_type, input_dimension]
        output_image_type = itk.Image[output_pixel_type, output_dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[input_image_type]
        writer_type = itk.ImageFileWriter[output_image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader_stack = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader_stack.SetImageIO(image_IO)

        ## Read images
        reader_stack.SetFileName(self.dir_input + filename + ".nii.gz")
        reader_stack.Update()

        ## Get image
        stack_itk = reader_stack.GetOutput()
        stack_sitk = sitk.ReadImage(dir_input + filename + ".nii.gz")

        ## Compute rotation
        rotation_itk = get_centered_rotation_itk(stack_itk, angles_in_deg)
        rotation_sitk = get_centered_rotation_sitk(stack_sitk, angles_in_deg)

        ## Fetch parameters
        parameters_itk = np.array(rotation_itk.GetParameters())
        fixed_parameters_itk = np.array(rotation_itk.GetFixedParameters())

        parameters_sitk = np.array(rotation_sitk.GetParameters())
        fixed_parameters_sitk = np.array(rotation_sitk.GetFixedParameters())

        ## Test correspondences
        self.assertEqual(np.around(
            np.linalg.norm(parameters_sitk - parameters_itk)
            , decimals = self.accuracy), 0 )

        self.assertEqual(np.around(
            np.linalg.norm(fixed_parameters_sitk - fixed_parameters_itk)
            , decimals = self.accuracy), 0 )


    def test_05_get_composited_itk_transform_2D(self):
        filename_2D = "BrainWeb_2D"

        angle_in_deg = 40

        ## Define types of input and output pixels and state dimension of images
        input_pixel_type = itk.D
        output_pixel_type = input_pixel_type

        input_dimension = 2
        output_dimension = input_dimension

        ## Define type of input and output image
        input_image_type = itk.Image[input_pixel_type, input_dimension]
        output_image_type = itk.Image[output_pixel_type, output_dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[input_image_type]
        writer_type = itk.ImageFileWriter[output_image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader_stack = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader_stack.SetImageIO(image_IO)

        ## Read images
        reader_stack.SetFileName(self.dir_input + filename_2D + ".nii.gz")
        reader_stack.Update()

        ## Get image
        stack_itk = reader_stack.GetOutput()
        stack_sitk = sitk.ReadImage(dir_input + filename_2D + ".nii.gz")

        ## Compute rotation
        rotation_itk = get_centered_rotation_itk(stack_itk, angle_in_deg)
        rotation_sitk = get_centered_rotation_sitk(stack_sitk, angle_in_deg)

        ## Get image transform
        transform_image_itk = get_itk_affine_transform_from_itk_image(stack_itk)
        transform_image_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(stack_sitk)

        ## Get composited transform
        transform_itk = get_composited_itk_affine_transform(rotation_itk, transform_image_itk)
        transform_sitk = sitkh.get_composited_sitk_affine_transform(rotation_sitk, transform_image_sitk)

        ## Fetch parameters
        parameters_itk = np.array(transform_itk.GetParameters())
        fixed_parameters_itk = np.array(transform_itk.GetFixedParameters())

        parameters_sitk = np.array(transform_sitk.GetParameters())
        fixed_parameters_sitk = np.array(transform_sitk.GetFixedParameters())

        ## Test correspondences
        self.assertEqual(np.around(
            np.linalg.norm(parameters_sitk - parameters_itk)
            , decimals = self.accuracy), 0 )

        self.assertEqual(np.around(
            np.linalg.norm(fixed_parameters_sitk - fixed_parameters_itk)
            , decimals = self.accuracy), 0 )


    def test_05_get_composited_itk_transform_3D(self):
        filename_3D = "fetal_brain_a"

        ## Define types of input and output pixels and state dimension of images
        input_pixel_type = itk.D
        output_pixel_type = input_pixel_type

        input_dimension = 3
        output_dimension = input_dimension

        ## Define type of input and output image
        input_image_type = itk.Image[input_pixel_type, input_dimension]
        output_image_type = itk.Image[output_pixel_type, output_dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[input_image_type]
        writer_type = itk.ImageFileWriter[output_image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader_stack = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader_stack.SetImageIO(image_IO)

        ## Read images
        reader_stack.SetFileName(self.dir_input + filename_3D + ".nii.gz")
        reader_stack.Update()

        ## Get image
        stack_itk = reader_stack.GetOutput()
        stack_sitk = sitk.ReadImage(dir_input + filename_3D + ".nii.gz")

        ## Define rigid trafo
        angle_x_in_deg = 20
        angle_y_in_deg = 40
        angle_z_in_deg = 90
        translation = (10,1,-10)
        center = (0,10,0)

        angles = np.array([angle_x_in_deg, angle_y_in_deg, angle_z_in_deg])*np.pi/180
        tmp_sitk = sitk.Euler3DTransform(center, angles[0], angles[1], angles[2], translation)

        matrix_sitk = np.array(tmp_sitk.GetMatrix()).reshape(3,3)
        center_sitk = np.array(tmp_sitk.GetFixedParameters())
        translation_sitk = np.array(tmp_sitk.GetTranslation())

        ## Construct rigid trafo as affine transform for itk and sitk
        rotation_sitk = sitk.AffineTransform(3)
        rotation_sitk.SetMatrix(matrix_sitk.flatten())
        rotation_sitk.SetCenter(center_sitk)
        rotation_sitk.SetTranslation(translation_sitk)

        rotation_itk = itk.AffineTransform[itk.D, 3].New()
        rotation_itk.SetMatrix(get_itk_matrix_from_numpy_array(matrix_sitk))
        rotation_itk.SetCenter(center_sitk)
        rotation_itk.SetTranslation(translation_sitk)

        ## Get image transform
        transform_image_itk = get_itk_affine_transform_from_itk_image(stack_itk)
        transform_image_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(stack_sitk)

        ## Get composited transform
        transform_itk = get_composited_itk_affine_transform(rotation_itk, transform_image_itk)
        transform_sitk = sitkh.get_composited_sitk_affine_transform(rotation_sitk, transform_image_sitk)

        ## Fetch parameters
        parameters_itk = np.array(transform_itk.GetParameters())
        fixed_parameters_itk = np.array(transform_itk.GetFixedParameters())

        parameters_sitk = np.array(transform_sitk.GetParameters())
        fixed_parameters_sitk = np.array(transform_sitk.GetFixedParameters())

        ## Test correspondences
        self.assertEqual(np.around(
            np.linalg.norm(parameters_sitk - parameters_itk)
            , decimals = self.accuracy), 0 )

        self.assertEqual(np.around(
            np.linalg.norm(fixed_parameters_sitk - fixed_parameters_itk)
            , decimals = self.accuracy), 0 )


    def test_06_get_transformed_image_2D(self):
        filename_2D = "BrainWeb_2D"

        angle_in_deg = 40

        ## Define types of input and output pixels and state dimension of images
        input_pixel_type = itk.D
        output_pixel_type = input_pixel_type

        input_dimension = 2
        output_dimension = input_dimension

        ## Define type of input and output image
        input_image_type = itk.Image[input_pixel_type, input_dimension]
        output_image_type = itk.Image[output_pixel_type, output_dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[input_image_type]
        writer_type = itk.ImageFileWriter[output_image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader_stack = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader_stack.SetImageIO(image_IO)

        ## Read images
        reader_stack.SetFileName(self.dir_input + filename_2D + ".nii.gz")
        reader_stack.Update()

        ## Get image
        stack_itk = reader_stack.GetOutput()
        stack_sitk = sitk.ReadImage(dir_input + filename_2D + ".nii.gz")

        ## Compute rotation
        rotation_itk = get_centered_rotation_itk(stack_itk, angle_in_deg)
        rotation_sitk = get_centered_rotation_sitk(stack_sitk, angle_in_deg)

        ## Get image transform
        transform_image_itk = get_itk_affine_transform_from_itk_image(stack_itk)
        transform_image_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(stack_sitk)

        ## Get composited transform
        transform_itk = get_composited_itk_affine_transform(rotation_itk, transform_image_itk)
        transform_sitk = sitkh.get_composited_sitk_affine_transform(rotation_sitk, transform_image_sitk)

        ## Warp image
        image_warped_itk = get_transformed_image(stack_itk, transform_itk)
        image_warped_sitk = sitkh.get_transformed_image(stack_sitk, transform_sitk)

        ## Fetch parameters
        direction_itk = get_numpy_array_from_itk_matrix(image_warped_itk.GetDirection())
        origin_itk = np.array(image_warped_itk.GetOrigin())
        spacing_itk = np.array(image_warped_itk.GetSpacing())

        direction_sitk = np.array(image_warped_sitk.GetDirection()).reshape(2,2)
        origin_sitk = np.array(image_warped_sitk.GetOrigin())
        spacing_sitk = np.array(image_warped_sitk.GetSpacing())

        ## Test correspondences
        self.assertEqual(np.around(
            np.linalg.norm(direction_sitk - direction_itk)
            , decimals = self.accuracy), 0 )

        self.assertEqual(np.around(
            np.linalg.norm(origin_sitk - origin_itk)
            , decimals = self.accuracy), 0 )

        self.assertEqual(np.around(
            np.linalg.norm(spacing_sitk - spacing_itk)
            , decimals = self.accuracy), 0 )


    def test_06_get_transformed_image_3D(self):
        filename_3D = "fetal_brain_a"

        ## Define types of input and output pixels and state dimension of images
        input_pixel_type = itk.D
        output_pixel_type = input_pixel_type

        input_dimension = 3
        output_dimension = input_dimension

        ## Define type of input and output image
        input_image_type = itk.Image[input_pixel_type, input_dimension]
        output_image_type = itk.Image[output_pixel_type, output_dimension]

        ## Define types of reader and writer
        reader_type = itk.ImageFileReader[input_image_type]
        writer_type = itk.ImageFileWriter[output_image_type]
        image_IO_type = itk.NiftiImageIO

        ## Instantiate reader and writer
        reader_stack = reader_type.New()
        writer = writer_type.New()

        ## Set image IO type to nifti
        image_IO = image_IO_type.New()
        reader_stack.SetImageIO(image_IO)

        ## Read images
        reader_stack.SetFileName(self.dir_input + filename_3D + ".nii.gz")
        reader_stack.Update()

        ## Get image
        stack_itk = reader_stack.GetOutput()
        stack_sitk = sitk.ReadImage(dir_input + filename_3D + ".nii.gz")

        ## Define rigid trafo
        angle_x_in_deg = 20
        angle_y_in_deg = 40
        angle_z_in_deg = 90
        translation = (10,1,-10)
        center = (0,10,0)

        angles = np.array([angle_x_in_deg, angle_y_in_deg, angle_z_in_deg])*np.pi/180
        tmp_sitk = sitk.Euler3DTransform(center, angles[0], angles[1], angles[2], translation)

        matrix_sitk = np.array(tmp_sitk.GetMatrix()).reshape(3,3)
        center_sitk = np.array(tmp_sitk.GetFixedParameters())
        translation_sitk = np.array(tmp_sitk.GetTranslation())

        ## Construct rigid trafo as affine transform for itk and sitk
        rotation_sitk = sitk.AffineTransform(3)
        rotation_sitk.SetMatrix(matrix_sitk.flatten())
        rotation_sitk.SetCenter(center_sitk)
        rotation_sitk.SetTranslation(translation_sitk)

        rotation_itk = itk.AffineTransform[itk.D, 3].New()
        rotation_itk.SetMatrix(get_itk_matrix_from_numpy_array(matrix_sitk))
        rotation_itk.SetCenter(center_sitk)
        rotation_itk.SetTranslation(translation_sitk)

        ## Get image transform
        transform_image_itk = get_itk_affine_transform_from_itk_image(stack_itk)
        transform_image_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(stack_sitk)

        ## Get composited transform
        transform_itk = get_composited_itk_affine_transform(rotation_itk, transform_image_itk)
        transform_sitk = sitkh.get_composited_sitk_affine_transform(rotation_sitk, transform_image_sitk)

        ## Warp image
        image_warped_itk = get_transformed_image(stack_itk, transform_itk)
        image_warped_sitk = sitkh.get_transformed_image(stack_sitk, transform_sitk)

        ## Fetch parameters
        direction_itk = get_numpy_array_from_itk_matrix(image_warped_itk.GetDirection())
        origin_itk = np.array(image_warped_itk.GetOrigin())
        spacing_itk = np.array(image_warped_itk.GetSpacing())

        direction_sitk = np.array(image_warped_sitk.GetDirection()).reshape(3,3)
        origin_sitk = np.array(image_warped_sitk.GetOrigin())
        spacing_sitk = np.array(image_warped_sitk.GetSpacing())

        ## Test correspondences
        self.assertEqual(np.around(
            np.linalg.norm(direction_sitk - direction_itk)
            , decimals = self.accuracy), 0 )

        self.assertEqual(np.around(
            np.linalg.norm(origin_sitk - origin_itk)
            , decimals = self.accuracy), 0 )

        self.assertEqual(np.around(
            np.linalg.norm(spacing_sitk - spacing_itk)
            , decimals = self.accuracy), 0 )


    ## Test APIs of itkOrientedGaussianInterpolateImageFunction in 2D. 
    # In particular, set/get sigma and set/get covariance
    def test_07_itkOrientedGaussianInterpolateImageFunction_2D_APIs(self):

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 2

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Create itk.OrientedGaussianInterpolateImageFunction
        filter_OrientedGaussian = filter_type.New()
        OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()

        ## Check set/get alpha 
        alpha = 1
        self.assertEqual(np.around(
            abs(OrientedGaussian.GetAlpha() - alpha)
            , decimals = self.accuracy), 0 )

        alpha = 2
        OrientedGaussian.SetAlpha(2)

        self.assertEqual(np.around(
            abs(OrientedGaussian.GetAlpha() - alpha)
            , decimals = self.accuracy), 0 )

        ## Check set/get Sigma and Covariance
        Cov = np.eye(dimension)
        Sigma = np.ones(dimension)

        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetSigma() - Sigma)
            , decimals = self.accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetCovariance() - Cov.flatten())
            , decimals = self.accuracy), 0 )

        ## Set covariance and sigma shall follow
        Cov[0,0] = 4
        Cov[0,1] = 1
        Cov[1,0] = Cov[0,1]
        Cov[1,1] = 9
        Sigma = np.sqrt(Cov.diagonal())
        OrientedGaussian.SetCovariance(Cov.flatten())

        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetSigma() - Sigma)
            , decimals = self.accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetCovariance() - Cov.flatten())
            , decimals = self.accuracy), 0 )

        ## Set sigma and covariance shall follow
        Sigma = np.array([3,4])
        Cov = np.zeros((dimension, dimension))
        Cov[0,0] = Sigma[0]**2
        Cov[1,1] = Sigma[1]**2
        OrientedGaussian.SetSigma(Sigma)

        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetSigma() - Sigma)
            , decimals = self.accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetCovariance() - Cov.flatten())
            , decimals = self.accuracy), 0 )

    ## Test APIs of itkOrientedGaussianInterpolateImageFunction in 3D. 
    # In particular, set/get sigma and set/get covariance
    def test_07_itkOrientedGaussianInterpolateImageFunction_3D_APIs(self):

        ## Define types of input and output pixels and state dimension of images
        pixel_type = itk.D
        dimension = 3

        ## Define type of input and output image
        image_type = itk.Image[pixel_type, dimension]

        ## Create itk.OrientedGaussianInterpolateImageFunction
        filter_OrientedGaussian = filter_type.New()
        OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()

        ## Check set/get alpha 
        alpha = 1
        self.assertEqual(np.around(
            abs(OrientedGaussian.GetAlpha() - alpha)
            , decimals = self.accuracy), 0 )

        alpha = 2
        OrientedGaussian.SetAlpha(2)

        self.assertEqual(np.around(
            abs(OrientedGaussian.GetAlpha() - alpha)
            , decimals = self.accuracy), 0 )

        ## Check set/get Sigma and Covariance
        Cov = np.eye(dimension)
        Sigma = np.ones(dimension)

        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetSigma() - Sigma)
            , decimals = self.accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetCovariance() - Cov.flatten())
            , decimals = self.accuracy), 0 )

        ## Set covariance and sigma shall follow
        Cov[0,0] = 4
        Cov[1,1] = 9
        Cov[1,1] = 16
        Cov[0,1] = 1
        Cov[0,2] = 2
        Cov[1,2] = 3
        Cov[1,0] = Cov[0,1]
        Cov[2,0] = Cov[0,2]
        Cov[2,1] = Cov[1,2]
        Sigma = np.sqrt(Cov.diagonal())
        OrientedGaussian.SetCovariance(Cov.flatten())

        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetSigma() - Sigma)
            , decimals = self.accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetCovariance() - Cov.flatten())
            , decimals = self.accuracy), 0 )

        ## Set sigma and covariance shall follow
        Sigma = np.array([3,4,5])
        Cov = np.zeros((dimension, dimension))
        Cov[0,0] = Sigma[0]**2
        Cov[1,1] = Sigma[1]**2
        Cov[2,2] = Sigma[2]**2
        OrientedGaussian.SetSigma(Sigma)

        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetSigma() - Sigma)
            , decimals = self.accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm(OrientedGaussian.GetCovariance() - Cov.flatten())
            , decimals = self.accuracy), 0 )


"""
Main Function
"""
if __name__ == '__main__':

    class Object(object):
        pass

    dir_input = "data/"
    dir_output = "results/"
    filename_HR_volume = "FetalBrain_reconstruction_4stacks"
    filename_stack = "FetalBrain_stack2_registered"
    filename_slice = "FetalBrain_stack2_registered_midslice"
    # filename_2D = "BrainWeb_2D"
    # filename_2D = "BrainWeb_2D_rotated"
    filename_2D = "SingleDot_2D_50"
    # filename_2D = "Cross_2D_50"

    # filename = "CTL_0_baseline_deleted_0.5"

    ## Define types of input and output pixels and state dimension of images
    pixel_type = itk.D
    input_pixel_type = pixel_type
    output_pixel_type = pixel_type

    input_dimension = 3
    output_dimension = input_dimension

    ## Define type of input and output image
    input_image_type = itk.Image[input_pixel_type, input_dimension]
    output_image_type = itk.Image[output_pixel_type, output_dimension]
    image_type_2D = itk.Image[itk.D, 2]

    ## Define types of reader and writer
    reader_type = itk.ImageFileReader[input_image_type]
    reader_type_2D = itk.ImageFileReader[image_type_2D]
    writer_type = itk.ImageFileWriter[output_image_type]
    writer_type_2D = itk.ImageFileWriter[image_type_2D]
    image_IO_type = itk.NiftiImageIO

    ## Instantiate reader and writer
    reader_HR_volume = reader_type.New()
    reader_stack = reader_type.New()
    reader_slice = reader_type.New()
    reader_2D = reader_type_2D.New()
    writer = writer_type.New()
    writer_2D = writer_type_2D.New()

    ## Set image IO type to nifti
    image_IO = image_IO_type.New()
    reader_HR_volume.SetImageIO(image_IO)
    reader_stack.SetImageIO(image_IO)
    reader_slice.SetImageIO(image_IO)
    reader_2D.SetImageIO(image_IO)

    ## Read images
    reader_HR_volume.SetFileName(dir_input + filename_HR_volume + ".nii.gz")
    reader_HR_volume.Update()
    
    reader_stack.SetFileName(dir_input + filename_stack + ".nii.gz")
    reader_stack.Update()

    reader_slice.SetFileName(dir_input + filename_slice + ".nii.gz")
    reader_slice.Update()

    reader_2D.SetFileName(dir_input + filename_2D + ".nii.gz")
    reader_2D.Update()

    ## Get image
    HR_volume_itk = reader_HR_volume.GetOutput()
    stack_itk = reader_stack.GetOutput()
    slice_itk = reader_slice.GetOutput()
    image_2D_itk = reader_2D.GetOutput()

    HR_volume_sitk = sitk.ReadImage(dir_input + filename_HR_volume + ".nii.gz", sitk.sitkFloat32)
    stack_sitk = sitk.ReadImage(dir_input + filename_stack + ".nii.gz", sitk.sitkFloat32)
    slice_sitk = sitk.ReadImage(dir_input + filename_slice + ".nii.gz", sitk.sitkFloat32)
    image_2D_sitk = sitk.ReadImage(dir_input + filename_2D + ".nii.gz", sitk.sitkFloat32)

    """
    'Real' start:
    """  
    # point = np.array([5,5,5])
    # center = np.array([3,3,3])
    # Sigma_PSF = get_PSF_covariance_matrix(stack_sitk)
    # Sigma = get_PSF_scaled_inverse_rotated_covariance_matrix(HR_volume_sitk, slice_sitk, Sigma_PSF)

    # compute_PSF_blurred_point(point, center, Sigma)

    # sigma_x2 = 1
    # sigma_y2 = 2
    # sigma_z2 = 3
    # Sigma = np.diag([sigma_x2,sigma_y2,sigma_z2])
    
    DIM = 3

    angle_in_deg = 20

    ## cutoff-distance
    alpha = 10

    ## Set covariance matrix
    sigma_x2 = 10
    sigma_y2 = 1
    sigma_z2 = 10


    # print_affine_transform(rotation_itk, "\n\nRotation 2D (ITK)")
    # sitkh.print_sitk_transform(rotation_sitk, "\nRotation 2D (SimpleITK)")

    # print_affine_transform(transform_image_itk, "Image Transform (ITK)")
    # sitkh.print_sitk_transform(transform_image_sitk, "\nImage Transform (SimpleITK)")

    # sitk.WriteImage(image_2D_rotated_sitk, dir_input + filename_2D + "_rotated.nii.gz")

    # s_interpolators = ['NearestNeighbour', 'Linear', 'Gaussian', 'OrientedGaussian']
    # interpolator_type = s_interpolators[2]

    if DIM == 2:

        Sigma_PSF = np.diag([sigma_x2,sigma_y2])

        rotation_itk = get_centered_rotation_itk(image_2D_itk, angle_in_deg)
        rotation_sitk = get_centered_rotation_sitk(image_2D_sitk, angle_in_deg)

        image_2D_rotated_itk = get_transformed_image(image_2D_itk, rotation_itk)
        image_2D_rotated_sitk = sitkh.get_transformed_image(image_2D_sitk, rotation_sitk)

        transform_image_itk = get_itk_affine_transform_from_itk_image(image_2D_rotated_itk)
        transform_image_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(image_2D_rotated_sitk)

        # Sigma_PSF =  get_PSF_covariance_matrix(image_2D_sitk)
        Sigma_aligned = get_PSF_rotated_covariance_matrix(image_2D_sitk, image_2D_rotated_sitk, Sigma_PSF)
        Sigma = get_PSF_scaled_inverse_rotated_covariance_matrix(image_2D_sitk, image_2D_rotated_sitk, Sigma_PSF)

        print("Sigma_PSF = \n%s" %Sigma_PSF)
        print("Sigma_PSF_oriented = \n%s" %Sigma_aligned)
        print("Sigma_PSF_oriented_aligned_scaled = \n%s" %Sigma)


        ## Resample Image Filter
        filter_type_2D = itk.ResampleImageFilter[image_type_2D, image_type_2D]
        filter_Gaussian = filter_type_2D.New()
        filter_OrientedGaussian = filter_type_2D.New()

        ## Set input image
        filter_Gaussian.SetInput(image_2D_rotated_itk)
        filter_Gaussian.SetOutputParametersFromImage(image_2D_rotated_itk)
        filter_Gaussian.Update()

        filter_OrientedGaussian.SetInput(image_2D_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(image_2D_rotated_itk)
        filter_OrientedGaussian.Update()

        ## Choose interpolator
        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[image_type_2D, pixel_type].New()
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(np.sqrt(Sigma_PSF.diagonal()))
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)

        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[image_type_2D, pixel_type].New()
        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetCovariance(Sigma_aligned.flatten())
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)

        ## Apply Filter
        filter_Gaussian.Update()
        filter_OrientedGaussian.Update()


        # show_itk_image(image_itk=filter_Gaussian.GetOutput())
        # show_itk_image(image_itk=filter_Gaussian.GetOutput(), overlay_itk=filter_OrientedGaussian.GetOutput())

        writer_2D.SetFileName(dir_output + "Interpolation_DiagCov_Gaussian.nii.gz")
        writer_2D.SetInput(filter_Gaussian.GetOutput())
        # writer_2D.Update()

        writer_2D.SetFileName(dir_output + "Interpolation_DiagCov_OrientedGaussian.nii.gz")
        writer_2D.SetInput(filter_OrientedGaussian.GetOutput())
        # writer_2D.Update()
        

    elif DIM == 3:
        # Sigma_PSF = np.diag([sigma_x2,sigma_y2,sigma_z2])

        Sigma_PSF =  get_PSF_covariance_matrix(slice_sitk)
        Sigma_aligned = get_PSF_rotated_covariance_matrix(HR_volume_sitk, slice_sitk, Sigma_PSF)
        Sigma = get_PSF_scaled_inverse_rotated_covariance_matrix(HR_volume_sitk, slice_sitk, Sigma_PSF)

        print("Sigma_PSF = \n%s" %Sigma_PSF)
        print("Sigma_PSF_oriented = \n%s" %Sigma_aligned)
        print("Sigma_PSF_oriented_aligned_scaled = \n%s" %Sigma)

        ## Resample Image Filter
        filter_type = itk.ResampleImageFilter[input_image_type, output_image_type]
        filter_Gaussian = filter_type.New()
        filter_OrientedGaussian = filter_type.New()

        ## Set input image
        filter_Gaussian.SetInput(slice_itk)
        filter_Gaussian.SetOutputParametersFromImage(slice_itk)
        filter_Gaussian.Update()

        filter_OrientedGaussian.SetInput(HR_volume_itk)
        # filter_OrientedGaussian.SetInput(stack_itk)
        filter_OrientedGaussian.SetOutputParametersFromImage(slice_itk)
        filter_OrientedGaussian.Update()

        ## Choose interpolator
        # interpolator_Gaussian = get_interpolator(
        #     'Gaussian',
        #     HR_volume_sitk=HR_volume_sitk,
        #     slice_sitk=slice_sitk)
        # filter_Gaussian.SetInterpolator(interpolator_Gaussian)

        # interpolator_OrientedGaussian = get_interpolator(
        #     'OrientedGaussian',
        #     HR_volume_sitk=HR_volume_sitk,
        #     slice_sitk=slice_sitk)
        # filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)

        ## Choose interpolator
        interpolator_Gaussian = itk.GaussianInterpolateImageFunction[input_image_type, pixel_type].New()
        interpolator_Gaussian.SetAlpha(alpha)
        interpolator_Gaussian.SetSigma(np.sqrt(Sigma_PSF.diagonal()))
        filter_Gaussian.SetInterpolator(interpolator_Gaussian)

        interpolator_OrientedGaussian = itk.OrientedGaussianInterpolateImageFunction[input_image_type, pixel_type].New()
        interpolator_OrientedGaussian.SetAlpha(alpha)
        interpolator_OrientedGaussian.SetCovariance(Sigma_aligned.flatten())
        filter_OrientedGaussian.SetInterpolator(interpolator_OrientedGaussian)

        ## Apply Filter
        filter_Gaussian.Update()
        filter_OrientedGaussian.Update()


        # show_itk_image(image_itk=filter_Gaussian.GetOutput(), overlay_itk=filter_OrientedGaussian.GetOutput())

        writer.SetFileName(dir_output + "Interpolation_DiagCov_Gaussian.nii.gz")
        writer.SetInput(filter_Gaussian.GetOutput())
        # writer.Update()

        writer.SetFileName(dir_output + "Interpolation_DiagCov_OrientedGaussian.nii.gz")
        writer.SetInput(filter_OrientedGaussian.GetOutput())
        # writer.Update()

    # angle_x_in_deg = 10
    # angle_y_in_deg = 90
    # angle_z_in_deg = 30

    # rotation_sitk = get_centered_rotation_sitk( HR_volume_sitk, (angle_x_in_deg, angle_y_in_deg, angle_z_in_deg) )
    # HR_volume_rotated_sitk = sitkh.get_transformed_image(HR_volume_sitk, rotation_sitk)

    # sitk.WriteImage(HR_volume_rotated_sitk, dir_input + filename_HR_volume + "_rotated.nii.gz")



    """
    Unit tests:
    """
    print("\nUnit tests:\n--------------")
    unittest.main()