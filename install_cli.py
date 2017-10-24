# -*- coding: utf-8 -*-
import sys
import os
import re

# from niftymic.definitions import DIR_CPP, DIR_CPP_BUILD
# Leads to error:
#     File "<string>", line 1, in <module>
#     File "whatever-path/niftymic/setup.py", line 23, in <module>
#     from install_cli import main as install_cli
#     File "install_cli.py", line 6, in <module>
#     from niftymic.definitions import DIR_CPP, DIR_CPP_BUILD
#     File "niftymic/definitions.py", line 4, in <module>
#     from pysitk.definitions import DIR_TMP
#     ImportError: No module named pysitk.definitions

DIR_ROOT = os.path.abspath(__file__)
DIR_CPP = os.path.join(DIR_ROOT, "niftymic", "cli")
DIR_CPP_BUILD = os.path.join(DIR_ROOT, "build", "cpp")


def main(prefix_environ="NIFTYMIC_"):

    # Add cmake arguments marked by prefix_environ
    pattern = prefix_environ + "(.*)"
    p = re.compile(pattern)
    environment_vars = {p.match(f).group(1): p.match(f).group(0)
                        for f in os.environ.keys() if p.match(f)}
    cmake_args = []
    for k, v in environment_vars.iteritems():
        cmake_args.append("-D %s=%s" % (k, os.environ[v]))
    cmake_args.append(DIR_CPP)

    # Create build-directory
    if not os.path.isdir(DIR_CPP_BUILD):
        os.makedirs(DIR_CPP_BUILD)

    # Change current working directory to build-directory
    os.chdir(DIR_CPP_BUILD)

    # Compile using cmake
    cmd = "cmake %s" % (" ").join(cmake_args)
    print(cmd)
    os.system(cmd)
    cmd = "make -j"
    print(cmd)
    os.system(cmd)

    return 0

if __name__ == "__main__":
    sys.exit(main())
