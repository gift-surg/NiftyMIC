## \file SimpleITKHelper.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015

## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

## Import modules from src-folder
# import SimpleITKHelper as sitkh


# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/22_Transforms.html
# AddTransform does not work! Python always crashes! Moreover, the composition
# of AddTransform is stack based, i.e. first in -- last applied. Wtf!?
#
# \param[in] sitk::simple::AffineTransform for inner and outer transform
def get_composited_sitk_affine_transform(transform_outer, transform_inner):
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


def print_rigid_transformation(rigid_registration_transform, text="rigid transformation"):
    dim = rigid_registration_transform.GetDimension()

    matrix = np.array(rigid_registration_transform.GetMatrix()).reshape(dim,dim)
    translation = np.array(rigid_registration_transform.GetTranslation())

    print("\n" + text + ":")
    print("matrix = \n" + str(matrix))
    # print("angle = " + str(np.arccos(matrix[0,0])*180/np.pi))
    print("translation = " + str(translation))
    # print("\n")

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


def show_sitk_image(image_sitk, filename_out="test"):
    dir_output = "/tmp/"
    sitk.WriteImage(image_sitk, dir_output + filename_out + ".nii.gz")
    # cmd = "fslview " + dir_output + filename_out + ".nii.gz & "
    cmd = "itksnap " \
            + "-g " + dir_output + filename_out + ".nii.gz " \
            "& "
    os.system(cmd)


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
