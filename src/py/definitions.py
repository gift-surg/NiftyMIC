import os
import sys

dir_root = os.path.abspath(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'])
dir_test = os.path.join(dir_root, 'test-data/')
dir_build_cpp = os.path.join(dir_root, 'build', 'cpp')
# dir_tmp = os.path.join(dir_root, 'tmp')
dir_tmp = "/tmp/"

# Linked executables
itksnap_exe = "itksnap"
fslview_exe = "fslview"
niftyview_exe = "NiftyView"
bet_exe = "bet"
reg_aladin_exe = "reg_aladin"
reg_f3d_exe = "reg_f3d"

ALLOWED_EXTENSIONS = ["nii.gz", "nii"]
REGEX_FILENAMES = "[A-Za-z0-9+-_]+"
REGEX_FILENAME_EXTENSIONS = "(" + "|".join(ALLOWED_EXTENSIONS) + ")"

info = {
    "name": "Volumetric MRI Reconstruction from 2D Slices "
    "in the Presence of Motion",
    "version": "0.1.0",
    "description": "",
    "web_info": "",
    "repository": {
        "type": "",
        "url": "https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction"
    },
    "authors": "Michael Ebner",
    "dependencies": {
        # requirements.txt file automatically generated using pipreqs.
        # "python" : "{0}/requirements.txt".format(dir_root)
        # pip install -r requirements.txt before running the
        # code.
    }
}
