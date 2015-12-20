import irtk
import SimpleITK as sitk

dir_in = "data/"
dir_out = "results/"
img_filename = "placenta_s"


img_irtk = irtk.imread( dir_in+img_filename+".nii.gz", dtype='float32' )
img_sitk = sitk.ReadImage(dir_in+img_filename+".nii.gz",sitk.sitkFloat32)

print img_irtk.origin()
print img_irtk.ImageToWorld([0,0,0])
print img_sitk.GetOrigin()