import SimpleITK as sitk
import numpy as np
import nibabel as nib 
import unittest

"""
Set variables
"""
## Specify data
dir_input = "data/"
dir_output = "results/"
filename =  "0"

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
Functions
"""
def get_sitk_origin_from_nib_image(image_nib):
    affine_nib = image_nib.affine
    A_nib = affine_nib[0:-1,0:-1]
    t_nib = affine_nib[0:-1,3]

    return R.dot(t_nib)


def get_sitk_direction_matrix_from_nib_image(image_nib):
    affine_nib = image_nib.affine
    A_nib = affine_nib[0:-1,0:-1]
    spacing_nib = np.array(image_nib.header.get_zooms())
    S_nib_inv = np.diag(1/spacing_nib)

    return R.dot(A_nib).dot(S_nib_inv)


def get_sitk_affine_matrix_from_sitk_image(image_sitk):
    spacing_sitk = np.array(image_sitk.GetSpacing())
    S_sitk = np.diag(spacing_sitk)
    A_sitk = np.array(image_sitk.GetDirection()).reshape(3,3)

    return A_sitk.dot(S_sitk).flatten()


def get_sitk_affine_translation_from_sitk_image(image_sitk):
    return np.array(image_sitk.GetOrigin())


def get_nib_affine_matrix_from_sitk_image(image_sitk):
    spacing_sitk = np.array(image_sitk.GetSpacing())
    S_sitk = np.diag(spacing_sitk)
    A_sitk = np.array(image_sitk.GetDirection()).reshape(3,3)

    return R.dot(A_sitk).dot(S_sitk)


def get_nib_translation_from_sitk_image(image_sitk):
    origin_sitk = image_sitk.GetOrigin()

    return R.dot(origin_sitk)


def build_affine_matrix_from_sitk_image(image_sitk):
    A = np.eye(4)
    A[0:-1,0:-1] = get_nib_affine_matrix_from_sitk_image(image_sitk)
    A[0:-1,3] = get_nib_translation_from_sitk_image(image_sitk)

    return A


def TransformIndexToPhysicalPoint_by_hand(image_sitk, point):
    origin_sitk = image_sitk.GetOrigin()
    spacing_sitk = np.array(image_sitk.GetSpacing())
    A_sitk = np.array(image_sitk.GetDirection()).reshape(3,3)

    return A_sitk.dot(point*spacing_sitk) + origin_sitk


def TransformIndexToPhysicalPoint_from_nib_header(image_nib, point):
    affine_nib = image_nib.affine
    A_nib = affine_nib[0:-1,0:-1]
    t_nib = affine_nib[0:-1,3]

    return R.dot(A_nib.dot(point) + t_nib)


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


def run_examples():
    print("\nExamples:\n--------------")

    # Check computation to obtain the rotation matrices:
    print("np.sum(abs(A_sitk - A(nib))) = " + 
        str(np.sum(abs(A_sitk - get_sitk_direction_matrix_from_nib_image(stack_nib)))))
    print("np.sum(abs(A_nib - A(sitk))) = " + 
        str(np.sum(abs(A_nib - get_nib_affine_matrix_from_sitk_image(stack_sitk)))))

    # Check computation to translation vectors:
    print("np.sum(abs(t_sitk - t(nib))) = " + 
        str(np.sum(abs(origin_sitk - get_sitk_origin_from_nib_image(stack_nib)))))
    print("np.sum(abs(t_nib - t(sitk))) = " + 
        str(np.sum(abs(t_nib - get_nib_translation_from_sitk_image(stack_sitk)))))

    # Check homogeneous matrix built from sitk file:
    print("np.sum(abs(affine_nib - affine(sitk)) = " + 
        str(np.sum(abs(affine_nib - build_affine_matrix_from_sitk_image(stack_sitk)))))

    point = np.array((0,0,0))
    point = np.array((100,50,30))

    # shape = np.array(stack_nib.shape)
    # point = shape-1


    print("\nTransformed point: " + str(point))
    tmp_1 = A_nib.dot(point) + t_nib
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
    def test_transformation_to_obtain_nib_matrix(self):
        self.assertEqual(np.around(
            np.sum(abs(A_nib - get_nib_affine_matrix_from_sitk_image(stack_sitk)))
            , decimals = accuracy), 0 )


    def test_transformation_to_obtain_nib_translation(self):
        self.assertEqual(np.around(
            np.sum(abs(t_nib - get_nib_translation_from_sitk_image(stack_sitk)))
            , decimals = accuracy), 0 )


    def test_transformation_to_obtain_nib_affine_matrix(self):
        self.assertEqual(np.around(
            np.sum(abs(affine_nib - build_affine_matrix_from_sitk_image(stack_sitk)))
            , decimals = accuracy), 0 )


    def test_transformation_to_obtain_sitk_direction_matrix(self):
        self.assertEqual(np.around(
            np.sum(abs(A_sitk - get_sitk_direction_matrix_from_nib_image(stack_nib)))
            , decimals = accuracy), 0 )


    def test_transformation_to_obtain_sitk_origin(self):
        self.assertEqual(np.around(
            np.sum(abs(origin_sitk - get_sitk_origin_from_nib_image(stack_nib)))
            , decimals = accuracy), 0 )


    """
    Test SimpleITK embedded functions by replicating them 'by hand'
    """
    def test_sitk_TransformIndexToPhysicalPoint_of_origin(self):
        point = (0,0,0)
        self.assertEqual(np.around(
            np.sum(abs(TransformIndexToPhysicalPoint_by_hand(stack_sitk,point) - stack_sitk.TransformIndexToPhysicalPoint(point)))
            , decimals = accuracy), 0 )


    def test_sitk_TransformIndexToPhysicalPoint_of_arbitrary_point(self):
        point = (100,50,30)
        self.assertEqual(np.around(
            np.sum(abs(TransformIndexToPhysicalPoint_by_hand(stack_sitk,point) - stack_sitk.TransformIndexToPhysicalPoint(point)))
            , decimals = accuracy), 0 )


    def test_sitk_TransformIndexToPhysicalPoint_of_origin_from_nib_header(self):
        point = (0,0,0)
        self.assertEqual(np.around(
            np.sum(abs(TransformIndexToPhysicalPoint_from_nib_header(stack_nib,point) - stack_sitk.TransformIndexToPhysicalPoint(point)))
            , decimals = accuracy), 0 )


    def test_sitk_TransformIndexToPhysicalPoint_of_arbitrary_point_from_nib_header(self):
        point = (100,50,30)

        self.assertEqual(np.around(
            np.sum(abs(TransformIndexToPhysicalPoint_from_nib_header(stack_nib,point) - TransformIndexToPhysicalPoint_by_hand(stack_sitk,point)))
            , decimals = accuracy-2), 0 )


    """
    Test the correct use of itk::simple::Tranform objects
    """
    def test_forward_affine_transformation_applied_on_origin(self):
        point = (0,0,0)

        A = get_sitk_affine_matrix_from_sitk_image(stack_sitk)
        t = get_sitk_affine_translation_from_sitk_image(stack_sitk)

        T_sitk = sitk.AffineTransform(A,t)

        self.assertEqual(np.around(
            # np.sum(abs(point - composite_transform.TransformPoint(point)))
            np.sum(abs(np.array(stack_sitk.TransformIndexToPhysicalPoint(point)) - T_sitk.TransformPoint(point)))
            , decimals = accuracy), 0 )


    def test_forward_affine_transformation_applied_on_arbitrary_point(self):
        shape = np.array(stack_sitk.GetSize())
        point = shape-1

        A = get_sitk_affine_matrix_from_sitk_image(stack_sitk)
        t = get_sitk_affine_translation_from_sitk_image(stack_sitk)

        T_sitk = sitk.AffineTransform(A,t)

        self.assertEqual(np.around(
            np.sum(abs(np.array(stack_sitk.TransformIndexToPhysicalPoint(point)) - T_sitk.TransformPoint(point)))
            , decimals = accuracy), 0 )


    def test_backward_forward_transformation_applied_on_origin(self):
        point = np.array((0,0,0))

        A = get_sitk_affine_matrix_from_sitk_image(stack_sitk)
        t = get_sitk_affine_translation_from_sitk_image(stack_sitk)

        T_sitk = sitk.AffineTransform(A,t)

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


    def test_backward_forward_transformation_applied_on_arbitrary_point(self):
        shape = np.array(stack_sitk.GetSize())
        point = shape-1
        # point = (point_np[0],point_np[1],point_np[2])

        A = get_sitk_affine_matrix_from_sitk_image(stack_sitk)
        t = get_sitk_affine_translation_from_sitk_image(stack_sitk)

        T_sitk = sitk.AffineTransform(A,t)
        T_sitk_inv = T_sitk.GetInverse()

        self.assertEqual(np.around(
            # np.sum(abs(point - composite_transform.TransformPoint(point)))
            np.sum(abs(point - T_sitk_inv.TransformPoint(stack_sitk.TransformIndexToPhysicalPoint(point))))
            , decimals = accuracy), 0 )


    """
    Test how to composite functions correctly in SimpleITK
    """
    def test_get_composited_sitk_affine_transform(self):
        shape = np.array(stack_sitk.GetSize())
        point = shape-1

        # Create first 3D rigid transformation randomly
        angle = np.random.random(3)*2*np.pi #random angles between 0 and 2 \pi
        translation = np.round(np.random.random(3)*np.min(shape-1))
        center = (0,0,0)

        T_0 = sitk.Euler3DTransform(center, angle[0], angle[1], angle[2], translation)  

        # Create second 3D rigid transformation randomly
        angle = np.random.random(3)*2*np.pi #random angles between 0 and 2 \pi
        translation = np.round(np.random.random(3)*np.min(shape-1))

        T_1 = sitk.Euler3DTransform(center, angle[0], angle[1], angle[2], translation)  

        # T_composited = T_1 o T_0:
        T_composited = get_composited_sitk_affine_transform(T_1,T_0)

        # Compute results both separated and composited
        res_separate = T_1.TransformPoint(T_0.TransformPoint(point))
        res_composited = T_composited.TransformPoint(point)

        self.assertEqual(np.around(
            # np.sum(abs(point - composite_transform.TransformPoint(point)))
            np.sum(abs(np.array(res_separate) - res_composited))
            , decimals = accuracy), 0 )


"""
Main Function
"""
if __name__ == '__main__':
    """
    Fetch data
    """
    ## Read image: SimpleITK
    stack_sitk = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat32)

    ## Read image: Nibabel
    stack_nib = nib.load(dir_input+filename+".nii.gz")

    ## Collect data: SimpleITK
    origin_sitk = stack_sitk.GetOrigin()
    spacing_sitk = np.array(stack_sitk.GetSpacing())
    A_sitk = np.array(stack_sitk.GetDirection()).reshape(3,3)

    ## Collect data: Nibabel
    affine_nib = stack_nib.affine
    A_nib = affine_nib[0:-1,0:-1]
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
