## \file PythonHelper.py
#  \brief
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Nov 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
# import itk
# import SimpleITK as sitk
import numpy as np
import contextlib
import matplotlib.pyplot as plt
import time
import datetime
from PIL import Image
# from datetime import timedelta

## Import modules
# import utilities.SimpleITKHelper as sitkh


##
#       Wait for <ENTER> to proceed the execution
# \date       2016-11-06 15:41:43+0000
#
def pause():
    programPause = raw_input("Press the <ENTER> key to continue ...")


##
# Exit/Terminate execution
# \date       2017-02-02 16:04:48+0000
#
def exit():
    sys.exit()


def killall_itksnap():
    os.system("killall ITK-SNAP")


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
# Shows single 2D/3D array or a list of 2D arrays.
# \date       2017-02-07 10:06:25+0000
#
# \param      nda         Either 2D/3D numpy array or list of 2D numpy arrays #
# \param      title       The title of the figure
# \param      cmap        Color map "Greys_r", "jet", etc.
# \param      colorbar    The colorbar
# \param      directory   In case given, figure will be saved to this directory
# \param      save_type   Filename extension of figure in case it is saved
# \param      fig_number  Figure number. If 'None' previous figure will not be
#                         closed
#                         
def show_arrays(nda, title="data", cmap="Greys_r", colorbar=False, directory=None, save_type="pdf", fig_number=1):

    ## Show list of 2D arrays slice by slice
    if type(nda) is list:
        _show_2D_array_list_array_by_array(nda, title=title, cmap=cmap, colorbar=colorbar, fig_number=fig_number, directory=directory, save_type=save_type)
    
    ## Show single 2D/3D array
    else:
        show_array(nda, title=title, cmap=cmap, colorbar=colorbar, directory=directory, save_type=save_type, fig_number=fig_number)


##
# Show single 2D or 3D array
# \date       2017-02-07 10:22:58+0000
#
# \param      nda         Single 2D or 3D numpy array
# \param      title       The title of the figure
# \param      cmap        Color map "Greys_r", "jet", etc.
# \param      colorbar    The colorbar
# \param      directory   In case given, figure will be saved to this directory
# \param      save_type   Filename extension of figure in case it is saved
# \param      fig_number  Figure number. If 'None' previous figure will not be
#                         closed
#                         
def show_array(nda, title="data", cmap="Greys_r", colorbar=False, directory=None, save_type="pdf", fig_number=None):

    ## Show single 2D array
    if len(nda.shape) is 2:
        _show_2D_array(nda, title=title, cmap=cmap, colorbar=colorbar, directory=directory, save_type=save_type, fig_number=fig_number)

    ## Show single 3D array
    elif len(nda.shape) is 3:
        _show_3D_array_slice_by_slice(nda, title=title, cmap=cmap, colorbar=colorbar, directory=directory, save_type=save_type, fig_number=fig_number)


##
# Shows data array and save it if desired
# \date       2016-11-07 21:29:13+0000
#
# \param      nda        Data array (only 2D so far)
# \param      title      The title of the figure
# \param      cmap       Color map "Greys_r", "jet", etc.
# \param      directory  In case given, figure will be saved to this directory
# \param      save_type  Filename extension of figure in case it is saved
#
def _show_2D_array(nda, title="data", cmap="Greys_r", colorbar=False, directory=None, save_type="pdf", fig_number=None):

    ## Plot figure
    fig = plt.figure(fig_number)
    fig.clf()
    plt.imshow(nda, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if colorbar:
        plt.colorbar()

    plt.show(block=False)
    
    ## If directory is given: Save 
    if directory is not None:    
        ## Create directory in case it does not exist already
        create_directory(directory)
        
        ## Save figure to directory
        _save_figure(fig, directory, title, save_type)


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
def _show_3D_array_slice_by_slice(nda3D_zyx, title="data", cmap="Greys_r", colorbar=False, directory=None, save_type="pdf", fig_number=None):

    shape = nda3D_zyx.shape
    N_slices = shape[0]

    ## Define the grid to arrange the slices
    grid = _get_grid_size(N_slices)

    ## Plot figure
    fig = plt.figure(fig_number)
    fig.clf()
    plt.suptitle(title)
    ctr = 1
    for i in range(0, N_slices):
        
        plt.subplot(grid[0], grid[1], ctr)
        plt.imshow(nda3D_zyx[i,:,:], cmap=cmap)
        plt.title(str(i))
        plt.axis('off')
        
        ctr += 1

    print("Slices of " + title + " are shown in separate window.")
    plt.show(block=False)

    ## If directory is given: Save 
    if directory is not None:    
        ## Create directory in case it does not exist already
        create_directory(directory)
        
        ## Save figure to directory
        _save_figure(fig, directory, title+"_slice_0_to_"+str(N_slices-1), save_type)


##
# Plot list of 2D numpy arrays next to each other
# \date       2016-11-06 02:02:36+0000
#
# \param      nda2D_list   List of 2D numpy data arrays
# \param      title        The title
# \param      cmap         Color map "Greys_r", "jet", etc.
# \param      colorbar     The colorbar
# \param      fig_number   The fig number
# \param      directory    The directory
# \param      save_type    The save type
# \param      axis_aspect  The axis aspect, Can be 'auto', 'equal'
#
# \return     { description_of_the_return_value }
#
def _show_2D_array_list_array_by_array(nda2D_list, title="image", cmap="Greys_r", colorbar=False, fig_number=None, directory=None, save_type="pdf", axis_aspect='auto'):

    shape = nda2D_list[0].shape
    N_slices = len(nda2D_list)

    if type(title) is not list:
        title = [title]

    ## Define the grid to arrange the slices
    grid = _get_grid_size(N_slices)

    ## Extract min and max value of arrays for same scaling
    value_min = np.min(nda2D_list[0])
    value_max = np.max(nda2D_list[0])
    for i in range(1, N_slices):
        value_min = np.min([value_min, np.min(nda2D_list[i])])
        value_max = np.max([value_max, np.max(nda2D_list[i])])

    # print("value_min = %.2f" %(value_min))
    # print("value_max = %.2f" %(value_max))
    ## Plot figure
    fig = plt.figure(fig_number)
    fig.clf()
    ctr = 1
    for i in range(0, N_slices):
        if ctr is 1:
            ax1 = plt.subplot(grid[0], grid[1], ctr)
            ax1.set_aspect(axis_aspect)
        else:
            ax2 = plt.subplot(grid[0], grid[1], ctr, sharex=ax1)
            ax2.set_aspect(axis_aspect)
        im = plt.imshow(nda2D_list[i], cmap=cmap, vmin=value_min, vmax=value_max)
        if len(title) is N_slices:
            plt.title(title[i])
        else:
            plt.title(title[0]+"_"+str(i))
        plt.axis('off')
        if colorbar:
            # add_axes([left, bottom, width, height])
            cax = fig.add_axes([0.92, 0.05, 0.01, 0.9])
            fig.colorbar(im, cax=cax)
        ctr += 1

    print("Slices of data arrays are shown in separate window.")
    plt.show(block=False)

    ## If directory is given: Save 
    if directory is not None:    
        ## Create directory in case it does not exist already
        create_directory(directory)
        
        filename = title[0]
        for i in range(1, N_slices):
            filename += "_" + title[i]


        ## Save figure to directory
        _save_figure(fig, directory, filename, save_type)


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
    if N_slices < 3:
        grid = (1, N_slices)
    elif N_slices > 2 and N_slices < 9:
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
# Saves a figure to given directory
# \date       2017-02-07 10:19:09+0000
#
# \param      fig        The fig
# \param      directory  The directory
# \param      filename   The filename
# \param      save_type  Filename extension of figure in case it is saved
#
def _save_figure(fig, directory, filename, save_type):

    ## Add backslash if not given
    if directory[-1] is not "/":
        directory += "/"
    
    fig.savefig(directory + filename + "." + save_type)
    print("Figure was saved to " + directory + filename + "." + save_type)


##
# Closes all pyplot figures.
# \date       2017-02-07 10:30:57+0000
#
def close_all_figures():
    plt.close('all')


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
    return datetime.timedelta(seconds=elapsed_time_sec)


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
def print_numpy_array(nda, title=None, precision=3, suppress=False):
    with printoptions(precision=precision, suppress=suppress):
        if title is not None:
            sys.stdout.write(title + " = ")
            sys.stdout.flush()
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
    print_debug_info("File " + str(directory + filename + "." + filename_extension) + " was created.")

def print_debug_info(text):
    # print("---- Debug info: ----")
    print("--- " + text)
    # print("---------------------------")


def print_title(text, symbol="*"):
    print_line_separator(symbol=symbol)
    print_subtitle(text,symbol=symbol)


def print_subtitle(text, symbol="*"):
    print(3*symbol + " " + text + " " + 3*symbol)


def print_line_separator(add_newline=True, symbol="*", length=99):
    if add_newline:
        print("\n")
    print(symbol*length)

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
def append_array_to_file(directory, filename, array, filename_extension="txt", format="%.10e", delimiter="\t"):
    file_handle = open(directory + filename + "." + filename_extension, "a")
    np.savetxt(file_handle, array, fmt=format, delimiter=delimiter)
    file_handle.close()
    print_debug_info("Array was appended to file " + str(directory + filename + "." + filename_extension) + ".")

##
# Execute and show command in command window.
# \date       2016-12-06 17:37:57+0000
#
# \param      cmd           The command
# \param      verbose  The show command
#
# \return     { description_of_the_return_value }
#
def execute_command(cmd, verbose=True):
    if verbose:
        print("")
        print("---- Executed command: ----")
        print(cmd)
        print("---------------------------")
        print("")
    os.system(cmd)


##
# Creates a directory on the HDD
# \date       2016-12-06 18:02:23+0000
#
# \param      directory     The directory
# \param      delete_files  The delete files
#
def create_directory(directory, delete_files=False):

    ## Add slash in case not existing
    if directory[-1] not in ["/"]:
        print directory
        directory += "/"
        print directory

    ## Create directory in case it does not exist already
    if not os.path.isdir(directory):
        os.system("mkdir -p " + directory)
        print_debug_info("Directory " + directory + " created.")

    if delete_files:
        clear_directory(directory)

##
# Clear all data in given directory
# \date       2017-02-02 16:47:15+0000
#
# \param      directory  The directory
#
def clear_directory(directory):
    os.system("rm -rf " + directory + "*")
    print_debug_info("All files in " + directory + " are removed.")


def get_current_date_and_time_strings():
    now = datetime.datetime.now()
    date = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2)
    time = str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

    return date, time


##
# Create a grid. Can be used to visualize deformation fields
# \date       2017-02-07 12:12:32+0000
#
# \param      shape     The shape
# \param      spacing   The spacing
# \param      value_fg  The value foreground
# \param      value_bg  The value background
#
# \return    image grid as numpy array
#
def create_image_grid(shape, spacing, value_bg=255, value_fg=0):

    nda = np.ones(shape)*value_bg

    for i in range(0, shape[0]):
        nda[i, 0::spacing] = value_fg

    for i in range(0, shape[1]):
        nda[0::spacing,i] = value_fg

    return nda


##
# Creates an image with slope.
# \date       2017-02-07 12:28:08+0000
#
# \param      shape     The shape
# \param      slope     The slope
# \param      value_fg  The value foreground
# \param      value_bg  The value background
# \param      offset    The offset
#
# \return     image with slope intensity as numpy array
#
def create_image_with_slope(shape, slope=1, value_bg=0, value_fg=255, offset=0):

    nda = np.ones(shape)*value_bg

    i = 0
    while i < nda.shape[0]:
        nda[i,:] = np.max([np.min([slope*i - offset, value_fg]),0])
        i = i+1

    return nda


##
# Creates an image pyramid.
# \date       2017-02-07 12:29:19+0000
#
# \param      shape     The shape
# \param      slope     The slope
# \param      value_fg  The value foreground
# \param      value_bg  The value background
#
# \return     { description_of_the_return_value }
#
def create_image_pyramid(length, slope=1, value_bg=0, value_fg=255, offset=(0,0)):

    shape = np.array([length,length])

    nda = np.ones(shape)*value_bg

    for i in range(0, nda.shape[0]/2):
        nda[i:-i,i:-i] = np.min([slope*i, value_fg])


    if np.abs(offset).sum() > 0:
        nda_offset = np.ones(shape)*value_bg
        nda_offset[offset[0]:, offset[1]:] = nda[0:-offset[0],0:-offset[1]]

        nda = nda_offset
    
    return nda


##
# Reads an image by using Image
# \date       2017-02-10 11:16:34+0000
#
# \param      filename  The filename including filename extension. E.g. 'png',
#                       'jpg'
#
# \return     Image data as numpy array
#
def read_image(filename):
        return np.asarray(Image.open(filename))


##
# Writes data array to image file
# \date       2017-02-10 11:18:21+0000
#
# \param      filename  The filename including filename extension
#
def write_image(nda, filename):
    im = Image.fromarray(nda)
    im.save(filename)
