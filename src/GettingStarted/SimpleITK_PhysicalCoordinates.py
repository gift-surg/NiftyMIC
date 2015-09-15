import SimpleITK as sitk
import numpy as np
import nibabel as nib 


dir_input = "data/"
dir_output = "results/"
filename =  "0"


## Read image: SimpleITK
stack_sitk = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat32)

## Read image: Nibabel
stack_nib = nib.load(dir_input+filename+".nii.gz")

## Collect data: SimpleITK
origin_sitk = stack_sitk.GetOrigin()
spacing_sitk = np.array(stack_sitk.GetSpacing())
A_sitk_GetDirection = np.array(stack_sitk.GetDirection()).reshape(3,3)

## Collect data: Nibabel
affine_nib = stack_nib.affine
A_nib = affine_nib[0:-1,0:-1]
t_nib = affine_nib[0:-1,3]

## Rotation matrix:
theta = np.pi
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
    ])

## Test
# point = (0,0,0)
point = (100,50,30)

print("\nTransformed point: " + str(point))

print("\nNifti-Header:")
print("Affine transformation (separately): " + str(
    A_nib.dot(point) + t_nib
    ))

print("\nSimpleITK:")
print("IndexToPhysicalPoint: " + str(stack_sitk.TransformIndexToPhysicalPoint(point)))
tmp = A_sitk_GetDirection.dot(point*spacing_sitk) + origin_sitk
print("GetDirection: " + str(tmp))
print("Rotation corrected (equal to 'Nifti-results'): " + str(R.dot(tmp)))

# print("Affine transformation (homogeneous): " + str(
#     affine_nib.dot(np.array([point[0], point[1], point[2],1]))
#     ))