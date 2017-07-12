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


##
# Error handling in case specified file does not exist
#
class FileNotExistent(Exception):

    ##
    # Store information on the missing file
    # \date       2017-06-29 12:49:18+0100
    #
    # \param      self          The object
    # \param      missing_file  string of missing file
    #
    def __init__(self, missing_file):
        self.missing_file = missing_file

    def __str__(self):
        error = "File '%s' does not exist" % (self.missing_file)
        return error


##
# Error handling in case specified directory does not exist
# \date       2017-07-11 17:02:12+0100
#
class DirectoryNotExistent(Exception):

    ##
    # Store information on the missing directory
    # \date       2017-07-11 17:02:46+0100
    #
    # \param      self            The object
    # \param      missing_directory  string of missing directory
    #
    def __init__(self, missing_directory):
        self.missing_directory = missing_directory

    def __str__(self):
        error = "Directory '%s' does not exist" % (self.missing_directory)
        return error


##
# Error handling in case multiple filenames exist
# (e.g. same filename but two different extensions)
# \date       2017-06-29 14:09:27+0100
#
class FilenameAmbiguous(Exception):

    ##
    # Store information on the ambiguous file
    # \date       2017-06-29 14:10:34+0100
    #
    # \param      self                The object
    # \param      ambiguous_filename  string of ambiguous file
    #
    def __init__(self, ambiguous_filename):
        self.ambiguous_filename = ambiguous_filename

    def __str__(self):
        error = "Filename '%s' ambiguous" % (self.ambiguous_filename)
        return error

##
# Error handling in case IO is not correct
# \date       2017-07-11 20:21:19+0100
#
class IOError(Exception):

    ##
    # Store information on the IO error
    # \date       2017-07-11 20:21:38+0100
    #
    # \param      self   The object
    # \param      error  The error
    #
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return self.error