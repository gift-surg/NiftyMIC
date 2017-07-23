#!/usr/bin/python

## \file DataAnonymizer.py
#  \brief 
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Dec 2016


## Import libraries
import string
import random
import string
import cPickle
import datetime
import os
import re               #regular expression

## Import modules
import pythonhelper.PythonHelper as ph

##
#
class DataAnonymizer(object):

    def __init__(self, dictionary=None, identifiers=None, prefix_identifiers="", filenames=None):
        
        self._dictionary = dictionary
        self._identifiers = identifiers
        self._prefix_identifiers = prefix_identifiers
        self._filenames = filenames

    ##
    # Generate identifiers
    # \date       2016-12-06 18:30:56+0000
    #
    # \param      self    The object
    # \param      length  The length
    #
    # \return     { description_of_the_return_value }
    #
    def generate_identifiers(self, randomized=False):
        
        if self._filenames is None:
            raise ValueError("Filenames are not set yet")

        ## Create random identifier based on string
        if randomized:

            ## Define amount of digits of random identifier
            digits = 4

            self._identifiers = [None]*len(self._filenames)
            for j in range(0, len(self._filenames)):
                self._identifiers[j] = ''.join(random.choice(string.ascii_uppercase + string.digits) for i in range(digits))

        ## Identifier based on alphabet
        else:
            ## ['a', 'b', 'c', ...]
            alphabet_str = list(string.ascii_lowercase)

            ## Set identifiers
            self._identifiers = alphabet_str[0:len(self._filenames)]

    ##
    # Sets/Gets the identifiers.
    # \date       2016-12-06 18:29:49+0000
    #
    def set_identifiers(self, identifiers):
        self._identifiers = identifiers

    def get_identifiers(self):
        return self._identifiers


    ##
    # Sets/Gets filenames
    # \date       2016-12-06 18:29:59+0000
    #
    def set_filenames(self, filenames):
        self._filenames = filenames

    def get_filenames(self):
        return self._filenames


    ##
    # Set/Get the identifier prefix
    # \date       2016-12-06 18:30:19+0000
    #
    def set_prefix_identifiers(self, prefix_identifiers):
        self._prefix_identifiers = prefix_identifiers

    def get_prefix_identifiers(self):
        return self._prefix_identifiers


    ##
    # Sets/Gets dictionary
    # \date       2016-12-06 18:29:59+0000
    #
    def set_dictionary(self, dictionary):
        self._dictionary = dictionary

    def get_dictionary(self):
        return self._dictionary

    ##
    # Generate a random dictionary based on given filenames and identifiers
    # \date       2016-12-06 18:33:32+0000
    #
    # \param      self  The object
    # \post       self._dictionary created
    #
    def generate_randomized_dictionary(self):

        self._dictionary = {}

        if len(self._filenames) is not len(self._identifiers):
            raise ValueError("Length of filenames does not match identifiers")

        ## Shuffle identifiers
        random.shuffle(self._identifiers)

        ## Create dictionary
        for i in range(0, len(self._filenames)):
            
            ## Update identifier including the prefix
            self._identifiers[i] = self._prefix_identifiers + self._identifiers[i]

            ## Create dictionary
            self._dictionary[self._identifiers[i]] = self._filenames[i]


    ##
    # Writes a dictionary.
    # \date       2016-12-06 19:26:22+0000
    #
    # \param      directory  The directory
    # \param      filename   The filename without extension
    #
    def write_dictionary(self, directory, filename, filename_backup=None, verbose=False):

        ## Save randomized dictionary
        f = open(directory + filename + ".p", 'wb')
        cPickle.dump(self._dictionary, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

        ## Write backup file (human readable)
        if filename_backup is None:
            filename_backup = filename + "_backup_human_readable"

        date, time = ph.get_current_date_and_time_strings()
        file_handle = open(directory + filename_backup + ".txt", "w")
        text = "## Randomized Dictionary " + date + " " + time + "\n"
        file_handle.write(text)
        file_handle.close()
        
        ## Print in an alphabetical order
        keys = sorted(self._dictionary.keys())
        for i in range(0, len(self._filenames)):
            file_handle = open(directory + filename_backup + ".txt", "a")
            text = keys[i] + " : " + self._dictionary[keys[i]] + "\n"
            file_handle.write(text)
            file_handle.close()
            if verbose:
                print("\t%s : %s" %(keys[i], self._dictionary[keys[i]]))

    ##
    # Reads a dictionary.
    # \date       2016-12-06 19:35:51+0000
    #
    # \param      self       The object
    # \param      directory  The directory
    # \param      filename   The filename without extension
    #
    def read_dictionary(self, directory, filename):
        
        ## Read dictionary
        f = open(directory + filename + ".p", 'rb')
        self._dictionary = cPickle.load(f)
        f.close()
        
        ## Retrieve identifiers and filenames
        self._identifiers = self._dictionary.keys()
        self._filenames = self._dictionary.values()


    ##
    # Print dictionary line by line
    # \date       2016-12-06 19:47:12+0000
    #
    # \param      self  The object
    #
    def print_dictionary(self):

        ## Print in an alphabetical order
        print("Content of current dictionary:")
        keys = sorted(self._dictionary.keys())
        for i in range(0, len(self._filenames)):
            print("\t%s : %s"%(keys[i], self._dictionary[keys[i]]))


    def anonymize_files(self, directory, filename_extension=".nii.gz"):

        for i in range(0, len(self._filenames)):
            filename_original = self._dictionary[self._identifiers[i]] + filename_extension
            filename_anonymized = self._identifiers[i] + filename_extension 

            if not os.path.isfile(directory + filename_original):
                print("%s: Nothing to anonymize" %(filename_original))
            else:

                cmd = "mv "
                cmd += directory + filename_original + " "
                cmd += directory + filename_anonymized + " "
                # print(cmd)
                os.system(cmd)
            

    ##
    # Reveals the anonymization and adds the original filename next to the
    # encryption.
    # \date       2016-12-06 20:27:23+0000
    #
    # \param      self                The object
    # \param      directory           The directory
    # \param      filename_extension  The filename extension
    #
    def reveal_anonymized_files(self, directory, filename_extension=".nii.gz"):

        for i in range(0, len(self._filenames)):
            filename_anonymized = self._identifiers[i] + filename_extension 
            filename_revealed = self._identifiers[i] + filename_extension + "_" + self._dictionary[self._identifiers[i]] + filename_extension

            if not os.path.isfile(directory + filename_anonymized):
                print("%s: Nothing to reveal" %(filename_anonymized))

            else:
                cmd = "mv "
                cmd += directory + filename_anonymized + " "
                cmd += directory + filename_revealed + " "
                # print(cmd)
                os.system(cmd)


    ##
    # Reveals the original filenames, assuming that 'reveal_anonymized_files'
    # has been run already
    # \date       2016-12-06 20:28:44+0000
    #
    # \param      self                The object
    # \param      directory           The directory
    # \param      filename_extension  The filename extension
    #
    def reveal_original_files(self, directory, filename_extension=".nii.gz"):

        for i in range(0, len(self._filenames)):
            filename_revealed = self._identifiers[i] + filename_extension + "_" + self._dictionary[self._identifiers[i]] + filename_extension
            filename_original = self._dictionary[self._identifiers[i]] + filename_extension 

            if not os.path.isfile(directory + filename_revealed):
                print("%s: Nothing to reveal" %(filename_revealed))

            else:
                cmd = "cp "
                cmd += directory + filename_revealed + " "
                cmd += directory + filename_original + " "
                # print(cmd)
                os.system(cmd)
