import os
import sys

dir_root = os.path.abspath(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'])
dir_test = os.path.join(dir_root, 'test-data/')
dir_build_cpp = os.path.join(dir_root, 'build' , 'cpp')

info = {
        "name": "Volumetric MRI Reconstruction from 2D Slices in the Presence of Motion",
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
                        # pip install -r requirements.txt before running the code.
                        }
        }
