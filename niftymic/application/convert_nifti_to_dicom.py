##
# \file convert_nifti_to_dicom.py
# \brief      Script to convert a 3D NIfTI image to DICOM.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Mar 2018
#

# NOTES (quite some candidates were tried to get a working solution):
#
# - nifti2dicom:
#   Although nifti2dicom allows the import of a DICOM header from a template
#   (-d) not all tags would be set correctly. E.g. if DOB is not given at
#   template, it would just be set to 01.01.1990 which would prevent the
#   resulting dcm file to be grouped correctly with the original data.
#   Moreover, annoying tags like 'InstitutionName' are set to their predefined
#   value which cannot be deleted (only overwritten).
#   Apart from that, only a relatively small selection of tags can be edited.
#   However, it does a good job in creating a series of 2D DICOM slices from a
#   NIfTI file (including correct image orientation!).
#
# - medcon:
#   A single 3D dcm file can be created but image orientation is flawed
#   when created from a nifti file directly.
#   However, if a 3D stack is created from a set of 2D dicoms, the orientation
#   stays correct.
#
# - pydicom:
#   Can only read a single 3D dcm file. In particular, it is not possible
#   to read a set of 2D slices unless a DICOMDIR is provided which is not
#   always guaranteed to exist (I tried to create it from 2D slices using
#   dcmmkdir from dcmtk and dcm4che -- neither seemed to work reliably)
#   Once the dicom file is read, pydicom does a really good job of updating
#   DICOM tags + there are plenty of tags available to be chosen from!
#   Saving a single 3D DICOM file is very easy too then.

import os
import pydicom

import pysitk.python_helper as ph

from niftymic.utilities.input_arparser import InputArgparser
from niftymic.definitions import DIR_TMP


COPY_DICOM_TAGS = {
    # important for grouping
    "PatientID",
    "PatientName",
    "PatientBirthDate",
    "StudyInstanceUID",

    # additional information
    "StudyID",
    "AcquisitionDate",
    "PatientSex",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturerModelName",
    "Modality",
    "StudyDescription",
}


def main():

    input_parser = InputArgparser(
        description="Convert obtained SRR from nifti to dicom format",
    )
    input_parser.add_filename(required=True)
    input_parser.add_option(
        option_string="--template",
        type=str,
        required=True,
        help="Template DICOM to extract relevant DICOM tags.",
    )
    input_parser.add_dir_output(required=True)
    input_parser.add_label(
        help="Label used for series description of DICOM output.",
        default="SRR")

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # Prepare for final DICOM output
    ph.create_directory(args.dir_output)
    path_to_output = os.path.join(args.dir_output, "%s.dcm" % args.label)

    # Prepare for intermediate output
    dir_output_2d_slices = os.path.join(DIR_TMP, "dicom_slices")
    ph.create_directory(dir_output_2d_slices, delete_files=True)

    # Create set of 2D DICOM slices from 3D NIfTI image
    # (correct image orientation!)
    ph.print_title("Create set of 2D DICOM slices from 3D NIfTI image")
    cmd_args = []
    cmd_args.append("-i %s" % args.filename)
    cmd_args.append("-o %s" % dir_output_2d_slices)
    cmd_args.append("-y")
    cmd = "nifti2dicom %s" % " ".join(cmd_args)
    ph.execute_command(cmd)

    # Combine set of 2D DICOM slices to form 3D DICOM image
    # (image orientation stays correct)
    ph.print_title("Combine set of 2D DICOM slices to form 3D DICOM image")
    cmd_args = []
    cmd_args.append("-f %s/*.dcm" % dir_output_2d_slices)
    cmd_args.append("-o %s" % path_to_output)
    cmd_args.append("-c dicom")
    cmd_args.append("-stack3d")
    cmd_args.append("-n")
    cmd_args.append("-qc")
    cmd_args.append("-w")
    cmd = "medcon %s" % " ".join(cmd_args)
    ph.execute_command(cmd)

    # Update all relevant DICOM tags accordingly
    ph.print_title("Update all relevant DICOM tags accordingly")
    print("")
    dataset = pydicom.dcmread(path_to_output)
    dataset_template = pydicom.dcmread(args.template)

    # Copy tags from template (to guarantee grouping with original data)
    update_dicom_tags = {}
    for tag in COPY_DICOM_TAGS:
        try:
            update_dicom_tags[tag] = getattr(dataset_template, tag)
        except:
            update_dicom_tags[tag] = ""

    # Additional tags
    update_dicom_tags["InstitutionName"] = "UCL, WEISS"
    update_dicom_tags["SeriesDescription"] = args.label
    update_dicom_tags["ImageComments"] = "*** NOT APPROVED ***"
    update_dicom_tags["AccessionNumber"] = "1"
    update_dicom_tags["SeriesNumber"] = "0"

    for tag in sorted(update_dicom_tags.keys()):
        value = update_dicom_tags[tag]
        setattr(dataset, tag, value)
        ph.print_info("%s: %s" % (tag, value))

    dataset.save_as(path_to_output)
    print("")
    ph.print_info("3D DICOM image written to %s" % path_to_output)

    return 0


if __name__ == '__main__':
    main()
