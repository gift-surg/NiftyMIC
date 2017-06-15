##
# \file Exceptions.py
# \brief      User-specific exceptions
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       June 2017
#


##
# Error handling in case the directory does not contain valid nifti file
# \date       2017-06-14 11:11:37+0100
#
class InputFilesNotValid(Exception):

    ##
    # \date       2017-06-14 11:12:55+0100
    #
    # \param      self       The object
    # \param      directory  Path to empty folder, string
    #
    def __init__(self, directory):
        self.directory = directory

    def __str__(self):
        error = "Folder '%s' does not contain valid nifti files." % (
            self.directory)
        return error


##
# Error handling in case of an attempted object access which is not being
# created yet
# \date       2017-06-14 11:20:33+0100
#
class ObjectNotCreated(Exception):

    ##
    # Store name of function which shall be executed to create desired object.
    # \date       2017-06-14 11:20:52+0100
    #
    # \param      self           The object
    # \param      function_call  function call missing to create the object
    # 
    def __init__(self, function_call):
        self.function_call = function_call

    def __str__(self):
        error = "Object has not been created yet. Run '%s' first." % (
            self.function_call)
        return error