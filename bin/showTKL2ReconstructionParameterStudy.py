#!/usr/bin/python

##
# \file showReconstructionParameterStudy.py
# \brief      Script to visualize the performed study reconstruction parameters for least squares
#             reconstruction with Tikhonov regularization.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2017
#

# Import libraries
import SimpleITK as sitk
import argparse
import numpy as np
import sys
import os

import pythonhelper.PythonHelper as ph
import numericalsolver.ReaderParameterStudy as ReaderParameterStudy

# Import modules
from volumetricreconstruction.utilities.InputArparser import InputArgparser


if __name__ == '__main__':

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    input_parser = InputArgparser(
        description="",
        prog="python " + os.path.basename(__file__),
    )

    input_parser.add_dir_input(required=True)
    input_parser.add_study_name(required=True)
    # input_parser.add_dir_output(required=True)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    parameter_study_reader = ReaderParameterStudy.ReaderParameterStudy(
        directory=args.dir_input, name=args.study_name)
    parameter_study_reader.read_study()

    measures = parameter_study_reader.get_measures()
    parameters_dic = parameter_study_reader.get_parameters()
    parameters_to_line_dic = parameter_study_reader.get_parameters_to_line()
    line_to_parameter_labels_dic = parameter_study_reader.\
        get_line_to_parameter_labels()

    # Get lines in result files associated to 'alpha'
    p = {k: (parameters_dic[k] if k == 'alpha' else parameters_dic[
             k][0]) for k in parameters_dic.keys()}
    lines = parameter_study_reader.get_lines_to_parameters(p)

    nda_data = parameter_study_reader.get_results("Data")
    nda_reg = parameter_study_reader.get_results("Reg")

    x = nda_data[lines, -1].flatten()
    y = nda_reg[lines, -1].flatten()
    labels = [line_to_parameter_labels_dic[i] for i in lines]
    
    ph.show_curves(y, x=x,
                   xlabel="Data",
                   ylabel="Regularizer",
                   labels=["alpha"],
                   markers=ph.MARKERS*100,
                   markevery=1,
                   # y_axis_style="loglog",
                   )