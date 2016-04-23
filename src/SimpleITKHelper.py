## \file SimpleITKHelper.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk
import itk
import numpy as np
import matplotlib.pyplot as plt

## Import modules from src-folder
# import SimpleITKHelper as sitkh


## AddTransform does not work! Python always crashes! Moreover, the composition
# of AddTransform is stack based, i.e. first in -- last applied. Wtf!?
# \param[in] sitk::simple::AffineTransform or EulerxDTransform for inner and outer transform
# \see http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/22_Transforms.html
def get_composited_sitk_affine_transform(transform_outer, transform_inner):
    
    ## Guarantee type sitk::simple::AffineTransform of transformations
    # transform_outer = sitk.AffineTransform(transform_outer)
    # transform_inner = sitk.AffineTransform(transform_inner)

    dim = transform_outer.GetDimension()

    A_inner = np.asarray(transform_inner.GetMatrix()).reshape(dim,dim)
    c_inner = np.asarray(transform_inner.GetCenter())
    t_inner = np.asarray(transform_inner.GetTranslation())

    A_outer = np.asarray(transform_outer.GetMatrix()).reshape(dim,dim)
    c_outer = np.asarray(transform_outer.GetCenter())
    t_outer = np.asarray(transform_outer.GetTranslation())

    A_composited = A_outer.dot(A_inner)
    c_composited = c_inner
    t_composited = A_outer.dot(t_inner + c_inner - c_outer) + t_outer + c_outer - c_inner

    return sitk.AffineTransform(A_composited.flatten(), t_composited, c_composited)


def get_sitk_image_direction_matrix_from_sitk_affine_transform(affine_transform_sitk, image_sitk):
    dim = len(image_sitk.GetSize())
    spacing_sitk = np.array(image_sitk.GetSpacing())
    S_inv_sitk = np.diag(1/spacing_sitk)

    A = np.array(affine_transform_sitk.GetMatrix()).reshape(dim,dim)

    return A.dot(S_inv_sitk).flatten()


def get_sitk_image_origin_from_sitk_affine_transform(affine_transform_sitk, image_sitk):
    """
    Important: Only tested for center=\0! Not clear how it shall be implemented,
            cf. Johnson2015a on page 551 vs page 107!

    Mostly outcome of application of get_composited_sitk_affine_transform and first transform_inner is image. 
    Therefore, center_composited is always zero on tested functions so far
    """
    dim = len(image_sitk.GetSize())

    affine_center = np.array(affine_transform_sitk.GetCenter())
    affine_translation = np.array(affine_transform_sitk.GetTranslation())
    
    R = np.array(affine_transform_sitk.GetMatrix()).reshape(dim,dim)

    return affine_center + affine_translation 
    # return affine_center + affine_translation - R.dot(affine_center)


def get_sitk_affine_matrix_from_sitk_image(image_sitk):
    dim = len(image_sitk.GetSize())
    spacing_sitk = np.array(image_sitk.GetSpacing())
    S_sitk = np.diag(spacing_sitk)
    A_sitk = np.array(image_sitk.GetDirection()).reshape(dim,dim)

    return A_sitk.dot(S_sitk).flatten()


def get_sitk_affine_translation_from_sitk_image(image_sitk):
    return np.array(image_sitk.GetOrigin())


def get_sitk_affine_transform_from_sitk_image(image_sitk):
    A = get_sitk_affine_matrix_from_sitk_image(image_sitk)
    t = get_sitk_affine_translation_from_sitk_image(image_sitk)

    return sitk.AffineTransform(A,t)


def get_3D_from_sitk_2D_rigid_transform(rigid_transform_2D_sitk):
    # Get parameters of 2D registration
    angle_z, translation_x, translation_y = rigid_transform_2D_sitk.GetParameters()
    center_x, center_y = rigid_transform_2D_sitk.GetFixedParameters()

    # Expand obtained translation to 3D vector
    translation_3D = (translation_x, translation_y, 0)
    center_3D = (center_x, center_y, 0)

    # Create 3D rigid transform based on 2D
    rigid_transform_3D = sitk.Euler3DTransform()
    rigid_transform_3D.SetRotation(0,0,angle_z)
    rigid_transform_3D.SetTranslation(translation_3D)
    rigid_transform_3D.SetFixedParameters(center_3D)
    
    return rigid_transform_3D


def get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D):
    ## Extract origin and direction matrix from slice:
    origin_3D = np.array(slice_3D.GetOrigin())
    direction_3D = np.array(slice_3D.GetDirection())

    ## Generate inverse transformations for translation and orthogonal transformations
    T_translation = sitk.AffineTransform(3)
    T_translation.SetTranslation(-origin_3D)

    T_rotation = sitk.AffineTransform(3)
    direction_inv = np.linalg.inv(direction_3D.reshape(3,3)).flatten()
    T_rotation.SetMatrix(direction_inv)

    ## T = T_rotation_inv o T_origin_inv
    T = get_composited_sitk_affine_transform(T_rotation,T_translation)

    return T


def get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D_sitk, T, slice_3D_sitk):
    ## Extract affine transformation to transform from Image to Physical Space
    T_PI = get_sitk_affine_transform_from_sitk_image(slice_3D_sitk)

    ## T = T_rotation_inv o T_origin_inv
    # T = get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D)
    # T = slice_3D.get_transform_to_align_with_physical_coordinate_system()
    T_inv = sitk.AffineTransform(T.GetInverse())

    ## T_PI_align = T_rotation_inv o T_origin_inv o T_PI: Trafo to align stack with physical coordinate system
    ## (Hence, T_PI_align(\i) = \spacing*\i)
    T_PI_align = get_composited_sitk_affine_transform(T, T_PI)

    ## Extract direction matrix and origin so that slice is oriented according to T_PI_align (i.e. with physical axes)
    # origin_PI_align = get_sitk_image_origin_from_sitk_affine_transform(T_PI_align,slice_3D)
    # direction_PI_align = get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_align,slice_3D)

    ## Extend to 3D rigid transform
    rigid_transform_3D = get_3D_from_sitk_2D_rigid_transform(rigid_transform_2D_sitk) 

    ## T_PI_in_plane_rotation_3D 
    ##    = T_origin o T_rotation o T_in_plane_rotation_2D_space 
    ##                      o T_rotation_inv o T_origin_inv o T_PI
    T_PI_in_plane_rotation_3D = sitk.AffineTransform(3)
    T_PI_in_plane_rotation_3D = get_composited_sitk_affine_transform(rigid_transform_3D, T_PI_align)
    T_PI_in_plane_rotation_3D = get_composited_sitk_affine_transform(T_inv, T_PI_in_plane_rotation_3D)

    return T_PI_in_plane_rotation_3D


## rigid_transform_*D (object type  Transform) as output of object sitk.ImageRegistrationMethod does not contain the
## member functions GetCenter, GetTranslation, GetMatrix whereas the objects sitk.Euler*DTransform does.
## Hence, create an instance sitk.Euler*D so that it can be used for composition of transforms as coded 
## in get_composited_sitk_affine_transform
def get_inverse_of_sitk_rigid_registration_transform(rigid_registration_transform):

    dim = rigid_registration_transform.GetDimension()

    if dim == 2:
        rigid_transform_2D = rigid_registration_transform

        ## Steps could have been chosen the same way as in the 3D case. However,
        ## here the computational steps more visible

        ## Extract parameters of 2D registration
        angle, translation_x, translation_y = rigid_transform_2D.GetParameters()
        center = rigid_transform_2D.GetFixedParameters()

        ## Create transformation used to align moving -> fixed

        ## Obtain inverse translation
        tmp_trafo = sitk.Euler2DTransform((0,0),-angle,(0,0))
        translation_inv = tmp_trafo.TransformPoint((-translation_x, -translation_y))

        ## Create instance of Euler2DTransform based on inverse = R_inv(x-c) - R_inv(t) + c
        return sitk.Euler2DTransform(center, -angle, translation_inv)

    elif dim == 3:
        rigid_transform_3D = rigid_registration_transform

        ## Create inverse transform of type Transform
        rigid_transform_3D_inv = rigid_transform_3D.GetInverse()

        ## Extract parameters of inverse 3D transform to feed them back to object Euler3DTransform:
        angle_x, angle_y, angle_z, translation_x, translation_y, translation_z = rigid_transform_3D_inv.GetParameters()
        center = rigid_transform_3D_inv.GetFixedParameters()

        ## Return inverse of rigid_transform_3D as instance of Euler3DTransform
        return sitk.Euler3DTransform(center, angle_x, angle_y, angle_z, (translation_x, translation_y, translation_z))
 
    


def check_sitk_mask_2D(mask_2D_sitk):

    mask_nda = sitk.GetArrayFromImage(mask_2D_sitk)

    if np.sum(mask_nda) > 1:
        return mask_2D_sitk

    else:
        mask_nda[:] = 1

        mask = sitk.GetImageFromArray(mask_nda)
        mask.CopyInformation(mask_2D_sitk)

        return mask


def get_transformed_image(image_init, transform):
    image = sitk.Image(image_init)
    
    affine_transform = get_sitk_affine_transform_from_sitk_image(image)

    transform = get_composited_sitk_affine_transform(transform, affine_transform)
    # transform = get_composited_sitk_affine_transform(get_inverse_of_sitk_rigid_registration_transform(affine_transform), affine_transform)

    direction = get_sitk_image_direction_matrix_from_sitk_affine_transform(transform, image)
    origin = get_sitk_image_origin_from_sitk_affine_transform(transform, image)

    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


## Read image from file and return as ITK obejct
#  \param[in] filename filename of image to read
#  \param[in] pixel_type itk pixel types, like itk.D, itk.F, itk.UC etc
#  \example read_itk_image("image.nii.gz", itk.D, 3) to read image stack
#  \example read_itk_image("mask.nii.gz", itk.UC, 3) to read image stack mask
def read_itk_image(filename, pixel_type=itk.D, dim=3):
    image_type = itk.Image[pixel_type, dim]
    # image_IO_type = itk.NiftiImageIO

    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(filename)
    # reader.SetImageIO(image_IO)

    reader.Update()
    image_itk = reader.GetOutput()
    image_itk.DisconnectPipeline()

    return image_itk


## Extract direction from SimpleITK-image so that it can be injected into
#  ITK-image
#  \param[in] image_sitk sitk.Image object
#  \return direction as itkMatrix object
def get_itk_direction_from_sitk_image(image_sitk):
    direction_sitk = image_sitk.GetDirection()

    return get_itk_direction_form_sitk_direction(direction_sitk)


## Convert direction from sitk.Image to itk.Image direction format
#  \param[in] direction_sitk direction obtained via GetDirection() of sitk.Image
#  \return direction which can be set as SetDirection() at itk.Image
def get_itk_direction_form_sitk_direction(direction_sitk):
    dim = np.sqrt(len(direction_sitk)).astype('int')
    m = itk.vnl_matrix_fixed[itk.D, dim, dim]()

    for i in range(0, dim):
        for j in range(0, dim):
            m.set(i,j,direction_sitk[dim*i + j])

    return itk.Matrix[itk.D, dim, dim](m)    


## Extract direction from ITK-image so that it can be injected into
#  SimpleITK-image
#  \param[in] image_itk itk.Image object
#  \return direction as 1D array of size dimension^2, np.array
def get_sitk_direction_from_itk_image(image_itk):
    direction_itk = image_itk.GetDirection()

    return get_sitk_direction_from_itk_direction(direction_itk)
    

## Convert direction from itk.Image to sitk.Image direction format
#  \param[in] direction_itk direction obtained via GetDirection() of itk.Image
#  \return direction which can be set as SetDirection() at sitk.Image
def get_sitk_direction_from_itk_direction(direction_itk):
    vnl_matrix = direction_itk.GetVnlMatrix()
    dim = np.sqrt(vnl_matrix.size()).astype('int')

    direction_sitk = np.zeros(dim*dim)
    for i in range(0, dim):
        for j in range(0, dim):
            direction_sitk[i*dim + j] = vnl_matrix(i,j)

    return direction_sitk


def get_sitk_Euler3DTransform_from_itk_Euler3DTransform(Euler3DTransform_itk):
    parameters_itk = Euler3DTransform_itk.GetParameters()
    fixed_parameters_itk = Euler3DTransform_itk.GetFixedParameters()
    
    N_params = parameters_itk.GetNumberOfElements()
    N_fixedparams = fixed_parameters_itk.GetNumberOfElements()
    
    parameters_sitk = np.zeros(N_params)
    fixed_parameters_sitk = np.zeros(N_fixedparams)

    for i in range(0, N_params):
        parameters_sitk[i] = parameters_itk.GetElement(i)

    for i in range(0, N_fixedparams):
        fixed_parameters_sitk[i] = fixed_parameters_itk.GetElement(i)


    Euler3DTransform_sitk = sitk.Euler3DTransform()
    Euler3DTransform_sitk.SetParameters(parameters_sitk)
    Euler3DTransform_sitk.SetFixedParameters(fixed_parameters_sitk)

    return Euler3DTransform_sitk

        

## Convert SimpleITK-image to ITK-image
#  \todo Check whether it is sufficient to just set origin, spacing and direction!
#  \param[in] image_sitk SimpleITK-image to be converted, sitk.Image object
#  \return converted image as itk.Image object
def convert_sitk_to_itk_image(image_sitk):

    ## Extract information ready to use for ITK-image
    dimension = image_sitk.GetDimension()
    origin = image_sitk.GetOrigin()
    spacing = image_sitk.GetSpacing()
    direction = get_itk_direction_from_sitk_image(image_sitk)
    nda = sitk.GetArrayFromImage(image_sitk)

    ## Define ITK image type according to pixel type of sitk.Object
    if image_sitk.GetPixelIDValue() is sitk.sitkFloat64:
        ## image stack
        image_type = itk.Image[itk.D, dimension]
    else:
        ## mask stack
        ## Couldn't use itk.UC (which apparently is used for masks normally)
        ## or any other "smaller format than itk.D" since 
        ## itk.MultiplyImageFilter[itk.UC, itk.D] does not work! (But
        ## which I need within InverseProblemSolver)
        # image_type = itk.Image[itk.D, dimension]
        # image_type = itk.Image[itk.UI, dimension]
        image_type = itk.Image[itk.UC, dimension]

    ## Create ITK image
    itk2np = itk.PyBuffer[image_type]
    image_itk = itk2np.GetImageFromArray(nda) 

    image_itk.SetOrigin(origin)
    image_itk.SetSpacing(spacing)
    image_itk.SetDirection(direction)

    image_itk.DisconnectPipeline()

    return image_itk


## Convert ITK-image to SimpleITK-image
#  \todo Check whether it is sufficient to just set origin, spacing and direction!
#  \param[in] image_itk ITK-image to be converted, itk.Image object
#  \return converted image as sitk.Image object
def convert_itk_to_sitk_image(image_itk):

    ## Extract information ready to use for SimpleITK-image
    dimension = image_itk.GetLargestPossibleRegion().GetImageDimension()
    origin = np.array(image_itk.GetOrigin())
    spacing = np.array(image_itk.GetSpacing())
    direction = get_sitk_direction_from_itk_image(image_itk)

    image_type = itk.Image[itk.D, dimension]
    itk2np = itk.PyBuffer[image_type]
    nda = itk2np.GetArrayFromImage(image_itk)

    ## Create SimpleITK-image
    image_sitk = sitk.GetImageFromArray(nda)

    image_sitk.SetOrigin(origin)
    image_sitk.SetSpacing(spacing)
    image_sitk.SetDirection(direction)

    return image_sitk



def print_rigid_transformation(rigid_registration_transform, text="rigid transformation"):
    dim = rigid_registration_transform.GetDimension()

    matrix = np.array(rigid_registration_transform.GetMatrix()).reshape(dim,dim)
    translation = np.array(rigid_registration_transform.GetTranslation())

    parameters = np.array(rigid_registration_transform.GetParameters())
    center = np.array(rigid_registration_transform.GetFixedParameters())

    print(text + ":")
    # print("matrix = \n" + str(matrix))
    # print("center = " + str(center))
    if isinstance(rigid_registration_transform, sitk.Euler3DTransform):
        print("\tangle_x, angle_y, angle_z = " + str(parameters[0:3]*180/np.pi) + " deg")

    else:
        print("\tangle = " + str(parameters[0]*180/np.pi) + " deg")
    print("\ttranslation = " + str(translation))

    return None


def plot_compare_sitk_2D_images(image0_2D_sitk, image1_2D_sitk, fig_number=1, flag_continue=0):

    fig = plt.figure(fig_number)
    plt.suptitle("intensity error norm = " + str(np.linalg.norm(sitk.GetArrayFromImage(image0_2D_sitk-image1_2D_sitk))))
    
    plt.subplot(1,3,1)
    plt.imshow(sitk.GetArrayFromImage(image0_2D_sitk), cmap="Greys_r")
    plt.title("image_0")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(sitk.GetArrayFromImage(image1_2D_sitk), cmap="Greys_r")
    plt.title("image_1")
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.imshow(sitk.GetArrayFromImage(image0_2D_sitk-image1_2D_sitk), cmap="Greys_r")
    plt.title("image_0 - image_1")
    plt.axis('off')

    ## Plot immediately or wait for following figures to come as well
    if flag_continue == 0:
        plt.show()
    else:
        plt.show(block=False)       # does not pause, but needs plt.show() at end 
                                    # of file to be visible
    return fig

## Show image with ITK-Snap. Image is saved to /tmp/ for that purpose
#  \param[in] image_sitk image to show
#  \param[in] segmentation 
#  \param[in] overlay image which shall be overlayed onto image_sitk (optional)
#  \param[in] title filename for file written to /tmp/ (optional)
def show_sitk_image(image_sitk, segmentation=None, overlay=None, title="test"):
    
    dir_output = "/tmp/"
    # cmd = "fslview " + dir_output + title + ".nii.gz & "

    if overlay is not None and segmentation is None:
        sitk.WriteImage(image_sitk, dir_output + title + ".nii.gz")
        sitk.WriteImage(overlay, dir_output + title + "_overlay.nii.gz")

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-o " + dir_output + title + "_overlay.nii.gz " \
            + "& "

    elif overlay is None and segmentation is not None:
        sitk.WriteImage(image_sitk, dir_output + title + ".nii.gz")
        sitk.WriteImage(segmentation, dir_output + title + "_segmentation.nii.gz")

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-s " + dir_output + title + "_segmentation.nii.gz " \
            + "& "

    elif overlay is not None and segmentation is not None:
        sitk.WriteImage(image_sitk, dir_output + title + ".nii.gz")
        sitk.WriteImage(segmentation, dir_output + title + "_segmentation.nii.gz")
        sitk.WriteImage(overlay, dir_output + title + "_overlay.nii.gz")

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-s " + dir_output + title + "_segmentation.nii.gz " \
            + "-o " + dir_output + title + "_overlay.nii.gz " \
            + "& "

    else:
        sitk.WriteImage(image_sitk, dir_output + title + ".nii.gz")

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "& "

    os.system(cmd)

    return None

## Show image with ITK-Snap. Image is saved to /tmp/ for that purpose
#  \param[in] image_itk image to show
#  \param[in] segmentation 
#  \param[in] overlay image which shall be overlayed onto image_itk (optional)
#  \param[in] title filename for file written to /tmp/ (optional)
def show_itk_image(image_itk, segmentation=None, overlay=None, title="test"):
    
    dir_output = "/tmp/"
    # cmd = "fslview " + dir_output + title + ".nii.gz & "

    dim = image_itk.GetBufferedRegion().GetImageDimension()

    image_type_seg = itk.Image[itk.UC, dim]
    writer_seg = itk.ImageFileWriter[image_type_seg].New()

    ## Define type of output image depending on what is used
    #  If type is not aligned with image_itk subsequent writer throws error
    try:
        ## Image type is 64 bit float for image stack
        pixel_type = itk.D
        image_type = itk.Image[pixel_type, dim]

        ## Define writer
        writer = itk.ImageFileWriter[image_type].New()
        # writer_2D.Update()
        # image_IO_type = itk.NiftiImageIO

        ## Write image_itk
        writer.SetInput(image_itk)
        writer.SetFileName(dir_output + title + ".nii.gz")
        writer.Update()
    except:
        ## Image type is unsigned char for mask stacks (works for mask related 
        #  operations like for itk.ImageMaskSpatialObject for registration)
        pixel_type = itk.UC
        image_type = itk.Image[pixel_type, dim]

        ## Define writer
        writer = itk.ImageFileWriter[image_type].New()
        # writer_2D.Update()
        # image_IO_type = itk.NiftiImageIO

        ## Write image_itk
        writer.SetInput(image_itk)
        writer.SetFileName(dir_output + title + ".nii.gz")
        writer.Update()


    if overlay is not None and segmentation is None:
        ## Write overlay:
        writer.SetInput(overlay)
        writer.SetFileName(dir_output + title + "_overlay.nii.gz")
        writer.Update()

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-o " + dir_output + title + "_overlay.nii.gz " \
            "& "
    
    elif overlay is None and segmentation is not None:
        ## Write segmentation:
        writer_seg.SetInput(segmentation)
        writer_seg.SetFileName(dir_output + title + "_segmentation.nii.gz")
        writer_seg.Update()

        cmd = "itksnap " \
            + "-g " + dir_output + title + ".nii.gz " \
            + "-s " + dir_output + title + "_segmentation.nii.gz " \
            + "& "

    elif overlay is not None and segmentation is not None:
        ## Write overlay:
        writer.SetInput(overlay)
        writer.SetFileName(dir_output + title + "_overlay.nii.gz")
        writer.Update()

        ## Write segmentation:
        writer_seg.SetInput(segmentation)
        writer_seg.SetFileName(dir_output + title + "_segmentation.nii.gz")
        writer_seg.Update()

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






