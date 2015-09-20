## \file SimpleITKHelper.py
#  \brief  
# 
#  \author Michael Ebner
#  \date September 2015

## Import libraries
# import os                       # used to execute terminal commands in python
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
# import SimpleITKHelper as sitkh


# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/22_Transforms.html
# AddTransform does not work! Python always crashes! Moreover, the composition
# of AddTransform is stack based, i.e. first in -- last applied. Wtf!?
#
# \param[in] sitk::simple::AffineTransform for inner and outer transform
def get_composited_sitk_affine_transform(transform_outer, transform_inner):
    A_inner = np.asarray(transform_inner.GetMatrix()).reshape(3,3)
    c_inner = np.asarray(transform_inner.GetCenter())
    t_inner = np.asarray(transform_inner.GetTranslation())

    A_outer = np.asarray(transform_outer.GetMatrix()).reshape(3,3)
    c_outer = np.asarray(transform_outer.GetCenter())
    t_outer = np.asarray(transform_outer.GetTranslation())

    A_composited = A_outer.dot(A_inner)
    c_composited = c_inner
    t_composited = A_outer.dot(t_inner + c_inner - c_outer) + t_outer + c_outer - c_inner

    return sitk.AffineTransform(A_composited.flatten(), t_composited, c_composited)


def get_3D_from_sitk_2D_rigid_transform(rigid_transform_2D):
    # Get parameters of 2D registration
    angle_z, translation_x, translation_y = rigid_transform_2D.GetParameters()
    center_x, center_y = rigid_transform_2D.GetFixedParameters()
    
    # Expand obtained translation to 3D vector
    translation_3D = (translation_x, translation_y, 0)
    center_3D = (center_x, center_y, 0)

    # Create 3D rigid transform based on 2D
    return sitk.Euler3DTransform(center_3D, 0, 0, angle_z, translation_3D)


def get_sitk_affine_matrix_from_sitk_image(image_sitk):
    spacing_sitk = np.array(image_sitk.GetSpacing())
    S_sitk = np.diag(spacing_sitk)
    A_sitk = np.array(image_sitk.GetDirection()).reshape(3,3)

    return A_sitk.dot(S_sitk).flatten()


def get_sitk_affine_translation_from_sitk_image(image_sitk):
    return np.array(image_sitk.GetOrigin())


def get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D, slice_3D):
    rigid_transform_3D = get_3D_from_sitk_2D_rigid_transform(rigid_transform_2D)

    A = get_sitk_affine_matrix_from_sitk_image(slice_3D)
    t = get_sitk_affine_translation_from_sitk_image(slice_3D)

    T_IP = sitk.AffineTransform(A,t)
    T_IP_inv = sitk.AffineTransform(T_IP.GetInverse())

    spacing = np.array(slice_3D.GetSpacing())
    S_inv_matrix = np.diag(1/spacing).flatten()
    S_inv = sitk.AffineTransform(S_inv_matrix,(0,0,0))

    # Trafo T = rigid_trafo_3D o T_IP_inv
    T = get_composited_sitk_affine_transform(rigid_transform_3D,T_IP_inv)

    # Trafo T = S_inv o rigid_trafo_3D o T_IP_inv
    # T = get_composited_sitk_affine_transform(S_inv,T)

    # Compute final composition T = T_IP o S_inv o rigid_trafo_3D o T_IP_inv
    return get_composited_sitk_affine_transform(T_IP,T)