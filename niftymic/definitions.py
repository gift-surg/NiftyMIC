import os
import sys

from pysitk.definitions import DIR_TMP
from pysitk.definitions import ITKSNAP_EXE
from pysitk.definitions import FSLVIEW_EXE
from pysitk.definitions import NIFTYVIEW_EXE

DIR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DIR_TEST = os.path.join(DIR_ROOT, "data", "tests")
DIR_TEMPLATES = os.path.join(DIR_ROOT, "data", "templates")
DIR_CPP_BUILD = os.path.join(DIR_ROOT, "build", "cpp")

ALLOWED_EXTENSIONS = ["nii.gz", "nii"]
REGEX_FILENAMES = "[A-Za-z0-9+-_]+"
REGEX_FILENAME_EXTENSIONS = "(" + "|".join(ALLOWED_EXTENSIONS) + ")"
