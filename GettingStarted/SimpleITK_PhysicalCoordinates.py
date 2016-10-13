import SimpleITK as sitk
import numpy as np
import nibabel as nib 
import unittest

"""
Functions
"""
def get_sitk_origin_from_nib_image(image_nib):
    affine_nib = image_nib.affine
    R_nib = affine_nib[0:-1,0:-1]
    t_nib = affine_nib[0:-1,3]

    return R.dot(t_nib)


def get_sitk_direction_matrix_from_nib_image(image_nib):
    affine_nib = image_nib.affine
    R_nib = affine_nib[0:-1,0:-1]
    spacing_nib = np.array(image_nib.header.get_zooms())
    S_nib_inv = np.diag(1/spacing_nib)

    return R.dot(R_nib).dot(S_nib_inv)


def get_sitk_affine_matrix_from_sitk_image(image_sitk):
    spacing_sitk = np.array(image_sitk.GetSpacing())
    S_sitk = np.diag(spacing_sitk)
    R_sitk = np.array(image_sitk.GetDirection()).reshape(3,3)

    return R_sitk.dot(S_sitk).flatten()


def get_sitk_affine_translation_from_sitk_image(image_sitk):
    """
    Important: conversion only for center=\0! (which is the case for sitk images)
    """
    origin_sitk = np.array(image_sitk.GetOrigin())
    # center_sitk = np.array(image_sitk.GetCenter())
    # R_sitk = np.array(image_sitk.GetDirection()).reshape(3,3)
    # return center_sitk + origin_sitk - R_sitk.dot(center_sitk)
    return origin_sitk


def get_sitk_affine_transform_from_sitk_image(image_sitk):
    A = get_sitk_affine_matrix_from_sitk_image(image_sitk)
    t = get_sitk_affine_translation_from_sitk_image(image_sitk)

    return sitk.AffineTransform(A,t)


def get_sitk_image_direction_from_sitk_affine_transform(affine_transform_sitk, image_sitk):
    spacing_sitk = np.array(image_sitk.GetSpacing())
    S_inv_sitk = np.diag(1/spacing_sitk)

    A = np.array(affine_transform_sitk.GetMatrix()).reshape(3,3)
    return A.dot(S_inv_sitk).flatten()


def get_sitk_image_origin_from_sitk_affine_transform(affine_transform_sitk, image_sitk):
    """
    Important: Only tested for center=\0! Not clear how it shall be implemented,
            cf. Johnson2015a on page 551 vs page 125!
    """
    affine_center = np.array(affine_transform_sitk.GetCenter())
    affine_translation = np.array(affine_transform_sitk.GetTranslation())
    
    # image_sitk_direction = get_sitk_affine_matrix_from_sitk_affine_transform(affine_transform_sitk, image_sitk) 

    # R = np.array(image_sitk_direction).reshape(3,3)

    return affine_center + affine_translation - R.dot(affine_center)
    # return affine_center + affine_translation - affine_transform_sitk.TransformPoint(affine_center)


def get_nib_orthogonal_matrix_from_sitk_image(image_sitk):
    spacing_sitk = np.array(image_sitk.GetSpacing())
    S_sitk = np.diag(spacing_sitk)
    R_sitk = np.array(image_sitk.GetDirection()).reshape(3,3)

    return R.dot(R_sitk).dot(S_sitk)


def get_nib_translation_from_sitk_image(image_sitk):
    origin_sitk = image_sitk.GetOrigin()

    return R.dot(origin_sitk)


def get_nib_affine_matrix_from_sitk_image(image_sitk):
    A = np.eye(4)
    A[0:-1,0:-1] = get_nib_orthogonal_matrix_from_sitk_image(image_sitk)
    A[0:-1,3] = get_nib_translation_from_sitk_image(image_sitk)

    return A


def TransformIndexToPhysicalPoint_by_hand(image_sitk, point):
    origin_sitk = image_sitk.GetOrigin()
    spacing_sitk = np.array(image_sitk.GetSpacing())
    R_sitk = np.array(image_sitk.GetDirection()).reshape(3,3)

    return R_sitk.dot(point*spacing_sitk) + origin_sitk


def TransformIndexToPhysicalPoint_from_nib_header(image_nib, point):
    affine_nib = image_nib.affine
    R_nib = affine_nib[0:-1,0:-1]
    t_nib = affine_nib[0:-1,3]

    return R.dot(R_nib.dot(point) + t_nib)


# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/22_Transforms.html
# AddTransform does not work! Python always crashes! Moreover, the composition
# of AddTransform is stack based, i.e. first in -- last applied. Wtf!?
#
# \param[in] sitk::simple::AffineTransform for inner and outer transform
def get_composite_sitk_affine_transform(transform_outer, transform_inner):
    A_inner = np.asarray(transform_inner.GetMatrix()).reshape(3,3)
    c_inner = np.asarray(transform_inner.GetCenter())
    t_inner = np.asarray(transform_inner.GetTranslation())

    A_outer = np.asarray(transform_outer.GetMatrix()).reshape(3,3)
    c_outer = np.asarray(transform_outer.GetCenter())
    t_outer = np.asarray(transform_outer.GetTranslation())

    A_composite = A_outer.dot(A_inner)
    c_composite = c_inner
    t_composite = A_outer.dot(t_inner + c_inner - c_outer) + t_outer + c_outer - c_inner

    return sitk.AffineTransform(A_composite.flatten(), t_composite, c_composite)


def run_examples():
    print("\nExamples:\n--------------")

    # Check computation to obtain the rotation matrices:
    print("np.sum(abs(R_sitk - R(nib))) = " + 
        str(np.sum(abs(R_sitk - get_sitk_direction_matrix_from_nib_image(stack_nib)))))
    print("np.sum(abs(R_nib - R(sitk))) = " + 
        str(np.sum(abs(R_nib - get_nib_orthogonal_matrix_from_sitk_image(stack_sitk)))))

    # Check computation to translation vectors:
    print("np.sum(abs(t_sitk - t(nib))) = " + 
        str(np.sum(abs(origin_sitk - get_sitk_origin_from_nib_image(stack_nib)))))
    print("np.sum(abs(t_nib - t(sitk))) = " + 
        str(np.sum(abs(t_nib - get_nib_translation_from_sitk_image(stack_sitk)))))

    # Check homogeneous matrix built from sitk file:
    print("np.sum(abs(affine_nib - affine(sitk)) = " + 
        str(np.sum(abs(affine_nib - get_nib_affine_matrix_from_sitk_image(stack_sitk)))))

    point = np.array((0,0,0))
    point = np.array((100,50,30))

    # shape = np.array(stack_nib.shape)
    # point = shape-1


    print("\nTransformed point: " + str(point))
    tmp_1 = R_nib.dot(point) + t_nib
    print("\nNifti-Header:\n--------------")
    print("Affine transformation (separately): " + str(tmp_1))
    print("Affine transformation (homogeneous): " + str(
        affine_nib.dot(np.array([point[0], point[1], point[2],1]))
        ))

    print("\nSimpleITK:\n--------------")
    print("IndexToPhysicalPoint: " + str(stack_sitk.TransformIndexToPhysicalPoint(point)))

    tmp = TransformIndexToPhysicalPoint_by_hand(stack_sitk,point)
    print("IndexToPhysicalPoint_by_hand: " + str(tmp))
    print("   np.sum(abs(IndexToPhysicalPoint - 'by hand')) = " 
        + str(np.sum(abs(stack_sitk.TransformIndexToPhysicalPoint(point)-tmp))))

    print("\nRotation corrected (equal to 'Nifti-results'): " + str(R.dot(tmp)))
    print("   np.sum(abs(diff)) = " + str(np.sum(abs(R.dot(tmp) - tmp_1))))


"""
Unit Test Class
"""
## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug accuracy, 2015
class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    """
    Test conversions between nibabel (i.e. Nifti header) and SimpleITK representations
    """
    def test_01_transformation_to_obtain_nib_matrix(self):
        self.assertEqual(np.around(
            np.sum(abs(R_nib - get_nib_orthogonal_matrix_from_sitk_image(stack_sitk)))
            , decimals = accuracy), 0 )


    def test_02_transformation_to_obtain_nib_translation(self):
        self.assertEqual(np.around(
            np.sum(abs(t_nib - get_nib_translation_from_sitk_image(stack_sitk)))
            , decimals = accuracy), 0 )


    def test_03_transformation_to_obtain_nib_affine_matrix(self):
        self.assertEqual(np.around(
            np.sum(abs(affine_nib - get_nib_affine_matrix_from_sitk_image(stack_sitk)))
            , decimals = accuracy), 0 )


    def test_04_transformation_to_obtain_sitk_direction_matrix(self):
        self.assertEqual(np.around(
            np.sum(abs(R_sitk - get_sitk_direction_matrix_from_nib_image(stack_nib)))
            , decimals = accuracy), 0 )


    def test_05_transformation_to_obtain_sitk_origin(self):
        self.assertEqual(np.around(
            np.sum(abs(origin_sitk - get_sitk_origin_from_nib_image(stack_nib)))
            , decimals = accuracy), 0 )


    """
    Test SimpleITK embedded functions by replicating them 'by hand'
    """
    def test_06_sitk_TransformIndexToPhysicalPoint_of_origin(self):
        point = (0,0,0)
        self.assertEqual(np.around(
            np.sum(abs(TransformIndexToPhysicalPoint_by_hand(stack_sitk,point) - stack_sitk.TransformIndexToPhysicalPoint(point)))
            , decimals = accuracy), 0 )


    def test_07_sitk_TransformIndexToPhysicalPoint_of_arbitrary_point(self):
        point = (100,50,30)
        self.assertEqual(np.around(
            np.sum(abs(TransformIndexToPhysicalPoint_by_hand(stack_sitk,point) - stack_sitk.TransformIndexToPhysicalPoint(point)))
            , decimals = accuracy), 0 )


    def test_08_sitk_TransformIndexToPhysicalPoint_of_origin_from_nib_header(self):
        point = (0,0,0)
        self.assertEqual(np.around(
            np.sum(abs(TransformIndexToPhysicalPoint_from_nib_header(stack_nib,point) - stack_sitk.TransformIndexToPhysicalPoint(point)))
            , decimals = accuracy), 0 )


    def test_09_sitk_TransformIndexToPhysicalPoint_of_arbitrary_point_from_nib_header(self):
        point = (100,50,30)

        self.assertEqual(np.around(
            np.sum(abs(TransformIndexToPhysicalPoint_from_nib_header(stack_nib,point) - TransformIndexToPhysicalPoint_by_hand(stack_sitk,point)))
            , decimals = accuracy-2), 0 )


    """
    Test the correct use of itk::simple::Tranform objects
    """
    def test_10_forward_affine_transformation_applied_on_origin(self):
        point = (0,0,0)

        T_sitk = get_sitk_affine_transform_from_sitk_image(stack_sitk)

        self.assertEqual(np.around(
            # np.sum(abs(point - composite_transform.TransformPoint(point)))
            np.sum(abs(np.array(stack_sitk.TransformIndexToPhysicalPoint(point)) - T_sitk.TransformPoint(point)))
            , decimals = accuracy), 0 )


    def test_11_forward_affine_transformation_applied_on_arbitrary_point(self):
        shape = np.array(stack_sitk.GetSize())
        point = shape-1

        T_sitk = get_sitk_affine_transform_from_sitk_image(stack_sitk)

        self.assertEqual(np.around(
            np.sum(abs(np.array(stack_sitk.TransformIndexToPhysicalPoint(point)) - T_sitk.TransformPoint(point)))
            , decimals = accuracy), 0 )


    def test_12_backward_forward_transformation_applied_on_origin(self):
        point = np.array((0,0,0))

        T_sitk = get_sitk_affine_transform_from_sitk_image(stack_sitk)
        T_sitk_inv = T_sitk.GetInverse()

        # composite_transform = sitk.Transform(T_sitk)
        # composite_transform = sitk.Transform(T_sitk_inv)
        # composite_transform = composite_transform.AddTransform(T_sitk_inv) #Python crashes!? => do it manually

        # print point
        # print stack_sitk.TransformIndexToPhysicalPoint(point)
        # print T_sitk.TransformPoint(point)
        # print T_sitk_inv.TransformPoint(stack_sitk.TransformIndexToPhysicalPoint(point))

        self.assertEqual(np.around(
            # np.sum(abs(point - composite_transform.TransformPoint(point)))
            np.sum(abs(point - T_sitk_inv.TransformPoint(stack_sitk.TransformIndexToPhysicalPoint(point))))
            , decimals = accuracy), 0 )


    def test_13_backward_forward_transformation_applied_on_arbitrary_point(self):
        shape = np.array(stack_sitk.GetSize())
        point = shape-1
        # point = (point_np[0],point_np[1],point_np[2])

        T_sitk = get_sitk_affine_transform_from_sitk_image(stack_sitk)
        T_sitk_inv = T_sitk.GetInverse()

        self.assertEqual(np.around(
            # np.sum(abs(point - composite_transform.TransformPoint(point)))
            np.sum(abs(point - T_sitk_inv.TransformPoint(stack_sitk.TransformIndexToPhysicalPoint(point))))
            , decimals = accuracy), 0 )


    """
    Test how to composite functions correctly in SimpleITK
    """
    def test_14_get_composite_sitk_affine_transform(self):
        shape = np.array(stack_sitk.GetSize())
        point = shape-1

        # Create first 3D rigid transformation randomly
        angle = np.random.random(3)*2*np.pi #random angles between 0 and 2 \pi
        translation = np.round(np.random.random(3)*np.min(shape-1))
        center = np.round(np.random.random(3)*np.min(shape-1))

        T_0 = sitk.Euler3DTransform(center, angle[0], angle[1], angle[2], translation)  

        # Create second 3D rigid transformation randomly
        angle = np.random.random(3)*2*np.pi #random angles between 0 and 2 \pi
        translation = np.round(np.random.random(3)*np.min(shape-1))
        center = np.round(np.random.random(3)*np.min(shape-1))

        T_1 = sitk.Euler3DTransform(center, angle[0], angle[1], angle[2], translation)  

        # T_composite = T_1 o T_0:
        T_composite = get_composite_sitk_affine_transform(T_1,T_0)

        # Compute results both separated and composite
        res_separate = T_1.TransformPoint(T_0.TransformPoint(point))
        res_composite = T_composite.TransformPoint(point)

        self.assertEqual(np.around(
            # np.sum(abs(point - composite_transform.TransformPoint(point)))
            np.sum(abs(np.array(res_separate) - res_composite))
            , decimals = accuracy), 0 )


    """
    Test conversion between itk::simple::AffineTransform and sitk_image
    """
    def test_15_conversion_between_sitk_affine_transform_and_sitk_image_data(self):
        stack_sitk_origin = np.array(stack_sitk.GetOrigin())
        stack_sitk_direction = np.array(stack_sitk.GetDirection()).reshape(3,3)

        """
        Important: Designed only for center=\0!
        """
        center = (0,0,0) 

        affine_matrix = get_sitk_affine_matrix_from_sitk_image(stack_sitk)
        affine_translation = get_sitk_affine_translation_from_sitk_image(stack_sitk)

        affine_transform = sitk.AffineTransform(affine_matrix, affine_translation, center)

        # Convert back to sitk image data (test valid in combination with test_10 and test_11):
        origin = get_sitk_image_origin_from_sitk_affine_transform(affine_transform, stack_sitk)
        direction = get_sitk_image_direction_from_sitk_affine_transform(affine_transform, stack_sitk)

        self.assertEqual(np.around(
            np.sum(abs(origin - stack_sitk_origin))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.sum(abs(np.array(direction).reshape(3,3) - stack_sitk_direction))
            , decimals = accuracy), 0 )

"""
Main Function
"""
if __name__ == '__main__':
    """
    Set variables
    """
    ## Specify data
    dir_input = "data/"
    dir_output = "results/"
    # filename =  "kidney_s"
    # filename =  "fetal_brain_a"
    # filename =  "fetal_brain_c"
    # filename =  "fetal_brain_s"
    filename =  "placenta_s"

    accuracy = 6 # decimal places for accuracy of unit tests

    ## Rotation matrix:
    # theta = np.pi
    # R = np.array([
    #     [np.cos(theta), -np.sin(theta), 0],
    #     [np.sin(theta), np.cos(theta), 0],
    #     [0, 0, 1]
    #     ])
    R = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
    
    """
    Fetch data
    """
    ## Read image: SimpleITK
    stack_sitk = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat64)

    ## Read image: Nibabel
    stack_nib = nib.load(dir_input+filename+".nii.gz")

    ## Collect data: SimpleITK
    origin_sitk = stack_sitk.GetOrigin()
    spacing_sitk = np.array(stack_sitk.GetSpacing())
    R_sitk = np.array(stack_sitk.GetDirection()).reshape(3,3)

    ## Collect data: Nibabel
    affine_nib = stack_nib.affine
    R_nib = affine_nib[0:-1,0:-1]
    t_nib = affine_nib[0:-1,3]


    """
    Examples
    """
    # run_examples()
    

    # A = get_sitk_affine_matrix_from_sitk_image(stack_sitk)
    # t = np.array(stack_sitk.GetOrigin())

    # # B = stack_nib.affine[0:-1,0:-1]
    # T_sitk = sitk.AffineTransform(A.reshape(-1),t)

    # T_sitk_inv = T_sitk.GetInverse()

    # composite_transform = sitk.Transform(T_sitk_inv)

    # shape = np.array(stack_nib.shape)

    # point = np.array(shape-1)
    # # point = (0,0,0)

    # print point
    # print stack_sitk.TransformIndexToPhysicalPoint(point)
    # print T_sitk.TransformPoint(point)
    # # print composit
    # print composite_transform.TransformPoint(stack_sitk.TransformIndexToPhysicalPoint(point))


    """
    Unit tests
    """
    print("\nUnit tests:\n--------------")
    unittest.main()
