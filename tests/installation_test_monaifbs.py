##
# \file installation_test_monaifbs.py
# \brief      Class to test installation of MONAIfbs
#             (https://github.com/gift-surg/MONAIfbs)
#
# \author     Marta Ranzini (marta.ranzini@kcl.ac.uk)
# \date       December 2020
#

import os
import unittest

import pysitk.python_helper as ph

from niftymic.definitions import DIR_TMP, DIR_TEMPLATES


class InstallationTestMonaiFbs(unittest.TestCase):

    def setUp(self):

        self.accuracy = 10

        self.path_to_image = os.path.join(DIR_TEMPLATES, "STA23.nii.gz")

    ##
    # Test whether monaifbs can be called
    # \date       2019-03-24 14:14:18+0000
    #
    def test_fetal_brain_seg(self):

        dir_output = os.path.join(DIR_TMP, "seg")
        cmd_args = ["niftymic_segment_fetal_brains"]
        cmd_args.append("--filenames %s" % self.path_to_image)
        cmd_args.append("--dir-output %s" % dir_output)
        # cmd_args.append("--verbose 1")
        cmd = " ".join(cmd_args)

        flag = ph.execute_command(cmd)
