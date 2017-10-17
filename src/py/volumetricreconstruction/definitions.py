import os
import sys

from pythonhelper.definitions import DIR_TMP
from pythonhelper.definitions import ITKSNAP_EXE
from pythonhelper.definitions import FSLVIEW_EXE
from pythonhelper.definitions import NIFTYVIEW_EXE

DIR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DIR_TEST = os.path.join(DIR_ROOT, "test-data/")
DIR_BUILD_CPP = os.path.join(DIR_ROOT, "build", "cpp")

ALLOWED_EXTENSIONS = ["nii.gz", "nii"]
REGEX_FILENAMES = "[A-Za-z0-9+-_]+"
REGEX_FILENAME_EXTENSIONS = "(" + "|".join(ALLOWED_EXTENSIONS) + ")"

BET_EXE = "bet"
