##
# \file ReadWriteMotionCorrectionResults.py
# \brief      Class to read and write motion corrected results
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
# 
# \remark needs to be very flexible! For the moment I stick to the file 


## Import libraries 
import SimpleITK as sitk
import numpy as np
import sys
import os

## Import modules
import base.Stack as st
import utilities.SimpleITKHelper as sitkh
import utilities.PythonHelper as ph
import utilities.FilenameParser as fp

##
class ReadWriteMotionCorrectionResults(object):

    def write_results_motion_correction(self, directory, filename, stack0, stack_corrected, slice_transforms, reference_image):

        print("Write results after motion correction")
        stack0.write(directory=directory, filename=filename+"_0", write_mask=True)
        stack_corrected.write(directory=directory, filename=filename+"_corrected", write_mask=True, write_slices=True)
        reference_image.write(directory=directory, filename="Ref_"+reference_image.get_filename(), write_mask=False, write_slices=False)

        for i in range(0, len(slice_transforms)):
            sitk.WriteTransform(slice_transforms[i], directory + filename + "_slicetransforms_" + str(i) + ".tfm")

        # np.savetxt(directory + filename + "_scale_inplane3D.txt", np.array([scale_inplane3D]))


    def read_results_motion_correction(self, directory, filename):

        stack0 = st.Stack.from_filename(dir_input=directory, filename=filename+"_0", suffix_mask=None)
        stack0.set_filename(filename)
        stack_corrected = st.Stack.from_slice_filenames(dir_input=directory, prefix_stack=filename+"_corrected", suffix_mask="_mask")

        slice_transforms_sitk = [None]*stack_corrected.sitk.GetDepth()
        for i in range(0, len(slice_transforms_sitk)):
            slice_transforms_sitk[i] = sitk.ReadTransform(directory + filename + "_slicetransforms_" + str(i) + ".tfm")
            slice_transforms_sitk[i] = sitk.AffineTransform(slice_transforms_sitk[i])

        # scale_inplane3D = np.loadtxt(directory + filename + "_scale_inplane3D.txt")

        ## Read PD reference used for motion correction
        filename_parser = fp.FilenameParser()
        filename_AB = filename_parser.get_separator_partitioned_filenames([filename], number_of_separators=1)
        # filename_reference_full = filename_parser.get_filenames_which_match_pattern_in_directory(directory, ["Ref_", filename_AB])[0]
        filename_reference_full = filename_parser.get_filenames_which_match_pattern_in_directory(directory, ["Ref_"])[0]
        filename_reference = filename_parser.replace_pattern([filename_reference_full], "Ref_")[0]
        reference_image = st.Stack.from_filename(directory, filename_reference_full)
        reference_image.set_filename(filename_reference)

        return stack0, stack_corrected, slice_transforms_sitk, reference_image
        