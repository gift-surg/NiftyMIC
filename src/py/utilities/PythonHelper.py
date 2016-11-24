## \file PythonHelper.py
#  \brief
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Nov 2016


## Import libraries
import os                       # used to execute terminal commands in python
# import sys
# import itk
# import SimpleITK as sitk
import numpy as np
import contextlib
import matplotlib.pyplot as plt
import time
from datetime import timedelta

## Import modules
# import utilities.SimpleITKHelper as sitkh


##
#       Wait for <ENTER> to proceed the execution
# \date       2016-11-06 15:41:43+0000
#
def pause():
    programPause = raw_input("Press the <ENTER> key to continue ...")


##
# Reads an input from the command line and returns it
# \date       2016-11-18 12:45:10+0000
#
# \param      infotext  The infotext
# \param      default   The default value which will be shown in square
#                       brackets
#
# \return     Input as either string, int or float, depending on what was
#             entered
#
def read_input(infotext="None", default=None):
    if default is None:
        text_in = raw_input(infotext + ": ")
        return text_in
    else:
        text_in = raw_input(infotext + " [" + str(default) + "]: ")

        if text_in in [""]:
            return default
        else:
            return text_in


##
# Plot data array and save it if desired
# \date       2016-11-07 21:29:13+0000
#
# \param      nda        Data array (only 2D so far)
# \param      title      The title of the figure
# \param      cmap       Color map "Greys_r", "jet", etc.
# \param      directory  In case given, figure will be saved to this directory
# \param      save_type  Filename extension of figure in case it is saved
#
# \remark     Could possibly use that as "master plot" and call the respective
#             function like 'plot_3D_array_slice_by_slice' depending on the
#             type
#
def plot_array(nda, title="data", cmap="Greys_r", colorbar=False, directory=None, save_type="pdf"):

    ## Plot figure
    fig = plt.figure(1)
    fig.clf()
    plt.imshow(nda, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if colorbar:
        plt.colorbar()

    ## If directory is given: Save 
    if directory is not None:
        
        ## Create directory in case it does not exist already
        os.system("mkdir -p " + directory)
        
        ## Add backslash if not given
        if directory[-1] is not "/":
            directory += "/"
        
        fig.savefig(directory + title + "." + save_type)
        print("Figure was saved to " + directory + title + "." + save_type)
    
    plt.show(block=False)


##
#       Plot 3D numpy array slice by slice next to each other
# \date       2016-11-06 01:39:28+0000
#
# All slices in the x-y-plane are plotted. The number of slices is given by the
# dimension in the z-axis.
#
# \param      nda3D_zyx  3D numpy data array in format (z,y,x) as it is given
#                        after sitk.GetArrayFromImage for instance
# \param      title      The title of the figure
# \param      cmap       Color map "Greys_r", "jet", etc.
#
def plot_3D_array_slice_by_slice(nda3D_zyx, title="image", cmap="Greys_r"):

    shape = nda3D_zyx.shape
    N_slices = shape[0]

    ## Define the grid to arrange the slices
    grid = _get_grid_size(N_slices)

    ## Plot figure
    fig = plt.figure(1)
    fig.clf()
    ctr = 1
    for i in range(0, N_slices):
        
        plt.subplot(grid[0], grid[1], ctr)
        plt.imshow(nda3D_zyx[i,:,:], cmap=cmap)
        plt.title(title+"_"+str(i))
        plt.axis('off')
        
        ctr += 1

    print("Slices of " + title + " are shown in separate window.")
    plt.show(block=False)


##
#       Plot list of 2D numpy arrays next to each other
# \date       2016-11-06 02:02:36+0000
#
# \param      nda2D_list  List of 2D numpy data arrays
# \param      title       The title
# \param      cmap        The cmap
#
def plot_2D_array_list(nda2D_list, title="image", cmap="Greys_r", colorbar=False):

    shape = nda2D_list[0].shape
    N_slices = len(nda2D_list)

    if type(title) is not list:
        title = [title]

    ## Define the grid to arrange the slices
    grid = _get_grid_size(N_slices)

    ## Plot figure
    fig = plt.figure(1)
    fig.clf()
    ctr = 1
    for i in range(0, N_slices):
        
        plt.subplot(grid[0], grid[1], ctr)
        plt.imshow(nda2D_list[i], cmap=cmap)
        if len(title) is N_slices:
            plt.title(title[i])
        else:
            plt.title(title[0]+"_"+str(i))
        plt.axis('off')
        if colorbar:
            plt.colorbar()
        ctr += 1

    print("Slices of data arrays are shown in separate window.")
    plt.show(block=False)


##
#       Gets the grid size given a number of 2D images
# \date       2016-11-06 02:02:20+0000
#
# \param      N_slices  The n slices
#
# \return     The grid size.
#
def _get_grid_size(N_slices):

    if N_slices > 40:
        raise ValueError("Too many slices to print")

    ## Define the view grid to arrange the slices
    if N_slices < 5:
        grid = (1, N_slices)
    elif N_slices > 4 and N_slices < 9:
        grid = (2, np.ceil(N_slices/2.).astype('int'))
    elif N_slices > 8 and N_slices < 13:
        grid = (3, np.ceil(N_slices/3.).astype('int'))
    elif N_slices > 12 and N_slices < 22:
        grid = (3, np.ceil(N_slices/3.).astype('int'))
    elif N_slices > 21 and N_slices < 29:
        grid = (4, np.ceil(N_slices/4.).astype('int'))
    else:
        grid = (5, np.ceil(N_slices/5.).astype('int'))

    return grid


##
#       Returns start time of execution
# \date       2016-11-06 17:15:00+0000
#
# \return     Start time of execution
#
def start_timing():
    return time.time()


##
#       Stops a timing and returns the time passed between given start
#             time.
# \date       2016-11-06 17:18:42+0000
#
# Conversion of elapsed time to 'reasonable' format,  i.e. hours, minutes,
# seconds, ... as appropriate.
#
# \param      start_time  The start time obtained via \p start_timing
#
# \return     Elapsed time as string
#
def stop_timing(start_time):
    end_time = time.time()
    elapsed_time_sec = end_time - start_time

    ## Convert to 'readable' format
    return timedelta(seconds=elapsed_time_sec)


##
# Print numpy array in certain format via \p printoptions below
# \date       2016-11-21 12:56:19+0000
# \see        http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
#
# \param      nda        numpy array
# \param      precision  Specifies the number of significant digits
# \param      suppress   Specifies whether or not scientific notation is
#                        suppressed for small numbers
#
def print_numpy_array(nda, precision=3, suppress=False):
    with printoptions(precision=precision, suppress=suppress):
        print(nda)

##
# Used in print_numpy_array to apply specific print formatting
# \see http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
#
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)


##
# Creates a file. Possibly existing file with the same name will be
# overwritten.
# \date       2016-11-24 12:16:50+0000
#
# \param      directory           The directory
# \param      filename            The filename
# \param      header              The header
# \param      filename_extension  The filename extension
#
def create_file(directory, filename, filename_extension="txt", header=""):
    file_handle = open(directory + filename + "." + filename_extension, "w")
    file_handle.write(header)
    file_handle.close()
    print("File " + str(directory + filename + "." + filename_extension) + " was created.")


##
# Appends an array to file.
# \date       2016-11-24 12:17:36+0000
#
# \param      directory           The directory
# \param      filename            The filename
# \param      array               The array
# \param      format              The format
# \param      delimiter           The delimiter
# \param      filename_extension  The filename extension
#
def append_array_to_file(directory, filename, array, filename_extension="txt", header="", format="%.10e", delimiter="\t"):
    file_handle = open(directory + filename + "." + filename_extension, "a")
    np.savetxt(file_handle, array, fmt=format, delimiter=delimiter)
    file_handle.close()
    print("Array was appended to file " + str(directory + filename + "." + filename_extension) + ".")
