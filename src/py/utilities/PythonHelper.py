## \file PythonHelper.py
#  \brief
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Nov 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import pickle
import numpy as np
import contextlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import datetime
from PIL import Image
# from datetime import timedelta
import itertools

from definitions import dir_tmp

## 
COLORS = [
    "r",        # red
    "b",        # blue
    "g",        # green
    "c",        # cyan
    "m",        # magenta
    "y",        # yellow
    "k",        # black
    "w",        # white
]
MARKERS = [
    "s",        # square
    "o",        # circle
    "v",        # triangle_down
    "x",        # x
    "p",        # pentagon
    "X",        # x (filled)
    "*",        # star
    "P",        # plus (filled)
    "^",        # triangle_up
    "<",        # triangle_left
    ">",        # triangle_right
    ".",        # point
    "+",        # plus
    "1",        # tri_down
    "2",        # tri_up
    "3",        # tri_left
    "4",        # tri_right
    "8",        # octagon
    ",",        # pixel
    "h",        # hexagon1
    "H",        # hexagon2
    "D",        # diamond
    "d",        # thin_diamond
    "|",        # vline
    "_",        # hline
    "None",     # nothing
    " ",        # nothing
    "",         # nothing
    '$...$',    # render the string using mathtext
]
LINESTYLES = [
    "-",        # solid line
    "--",       # dashed line
    "-.",       # dash-dotted line
    ":",        # dotted line
    "None",     # draw nothing
    " ",        # draw nothing
    "",         # draw nothing
]

##
# Writes variables.
# \date       2017-02-18 16:51:31+0000
#
# \param      variables  List of variables to store
# \param      directory  The directory
# \param      filename   The filename without extension
#
def write_variables(variables, directory, filename, filetype=".pckl"):

    directory = create_directory(directory)
    filename_out = directory + filename + filetype

    f = open(filename_out, 'wb')
    pickle.dump(variables, f, protocol=-1) #protocol=-1 for large files
    flag = f.close()

    if not flag:
        print_debug_info("Variables successfully written to " + filename_out )
    else:
        print_debug_info("Error: Variables could not be written")


##
# Reads variables.
# \date       2017-02-18 16:55:35+0000
#
# \param      directory  The directory
# \param      filename   The filename
#
# \return     List of read variables
#
def read_variables(directory, filename, filetype=".pckl"):
    
    filename_in = directory + filename + filetype

    f = open(filename_in, 'rb')
    variables = pickle.load(f)
    flag = f.close()

    if not flag:
        print_debug_info("Variables successfully read from " + filename_in )
    else:
        print_debug_info("Error: Variables could not be read")

    return variables


def get_function_call(function_name, args):
    cmd = "python " + function_name + " \\\n"

    for arg in sorted(vars(args)):
        argument = ("%s=" % (arg)).replace("_","-")
        cmd += "\t--" + argument + "%s" % (getattr(args, arg)) + " \\\n"

    return cmd

def write_function_call_to_executable_file(function_call, filename):
    
    call = "#!/bin/sh\n\n" + function_call

    text_file = open(filename, "w")
    text_file.write("%s" % call)
    text_file.close()
    print_debug_info("File " + filename + " generated.")

    # Make file executable
    os.system("chmod +x " + filename)

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


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
##
# { function_description }
# \date       2017-02-18 04:21:27+0000
#
# \param      y                 { parameter_description }
# \param      x                 { parameter_description }
# \param      xlabel            The xlabel
# \param      ylabel            The ylabel
# \param      title             The title
# \param      y_axis_style      The y axis style ("plot", "semilogy")
# \param      labels            The label
# \param      label_location    The label location
# \param      color             The color
# \param      markers           The marker
# \param      markevery         The markevery
# \param      linestyle         The linestyle
# \param      label_shadow      The label shadow
# \param      label_frameon     The label frameon
# \param      linewidth         The linewidth
# \param      markerfacecolors  The markerfacecolor ("none", None)
# \param      markersize        The markersize
# \param      fontfamily        The fontfamily ("serif", "sans-serif", "monospace")
# \param      fontname          The fontname ("Arial", "Times New Roman")
# \param      use_tex           The use tex
# \param      fontsize          The fontsize
# \param      backgroundcolor   The backgroundcolor
# \param      aspect_ratio      The aspect ratio ("auto", "equal")
# \param      save_figure       The save fig
# \param      directory         The directory
# \param      filename          The filename including extension
# \param      fig_number        The fig number
#
# \return     figure
#
def show_curves(y, x=None, xlabel="", ylabel="", title="", xlim=None, ylim=None, y_axis_style="plot", labels=None, label_location="upper right", color=None, markers=None, markevery=10, linestyle=None, label_shadow=False, label_frameon=True, label_fontsize=None, label_boundingboxtoanchor=None, label_ncol=1, linewidth=1, markerfacecolors=None, markersize=10, fontfamily="serif", fontname="Arial", use_tex=False, fontsize=12, backgroundcolor="None", aspect_ratio="auto", save_figure=False, directory=dir_tmp, filename="figure.pdf", fig_number=None, show_compact=0, subplots_left=0.08, subplots_bottom=0.11, subplots_right=0.99, subplots_top=0.83, subplots_wspace=0, subplots_hspace=0, figuresize=None):

    if type(y) is not list:
        y = [y]
    N_curves = len(y)

    ## Change font
    font = {
        "family"        :   fontfamily,
        fontfamily      :   fontname,
        "size"          :   fontsize,
        }
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=use_tex)

    if x is None:
        x = [None]*N_curves
        for i in range(N_curves):
            x[i] = np.arange(y[i].size)+1
    elif type(x) is not list:
        x = [x] * N_curves

    if type(linewidth) is not list and not isinstance(linewidth, np.ndarray):
        linewidth = [linewidth]*N_curves

    if type(labels) is not list:
        labels = [labels]*N_curves


    if figuresize is None:
        fig = plt.figure(fig_number)
    else:
        fig = plt.figure(fig_number, figsize=figuresize[::-1])
    
    if show_compact:
        plt.subplots_adjust(wspace=subplots_wspace, hspace=subplots_hspace, left=subplots_left, right=subplots_right, bottom=subplots_bottom, top=subplots_top)
    fig.clf()   

    ax = fig.add_subplot(111)
    for i in range(0,N_curves):

        ## Print line with preferred y axis style
        lines = eval("plt." + y_axis_style + "(x[i], y[i], label=labels[i])")
        # lines = plt.plot(x[i], y[i], label=labels[i])
        
        ## Extract line object to adjust line settings
        line = lines[0]
        
        ## Specify line settings
        line.set_linewidth(linewidth[i])
        if color is not None:
            line.set_color(color[i])
        if linestyle is not None:
            line.set_linestyle(linestyle[i])
        if markers is not None:
            line.set_marker(markers[i])
            line.set_markevery(markevery)
            line.set_markerfacecolor(markerfacecolors)
            line.set_markersize(markersize)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ## Add legend
    # legend = plt.legend(loc=label_location, shadow=label_shadow, frameon=label_frameon)
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(flip(handles, 2), flip(labels, 2), loc="lower center", shadow=label_shadow, frameon=label_frameon, bbox_to_anchor=(0.5, 1), ncol=N_curves/2)

    if label_fontsize is None:
        label_fontsize = fontsize

    # legend = plt.legend(loc="lower center", shadow=label_shadow, frameon=label_frameon, bbox_to_anchor=(0.47, 1), ncol=4, fontsize=label_fontsize)
    legend = plt.legend(loc=label_location, shadow=label_shadow, frameon=label_frameon, bbox_to_anchor=label_boundingboxtoanchor, ncol=label_ncol, fontsize=label_fontsize)
    
    ax.set_aspect(aspect_ratio)
    ax.set_facecolor(backgroundcolor)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show(block=False)
    

    if save_figure:
        ## Create directory in case it does not exist already
        create_directory(directory)

        ## Save figure to directory
        _save_figure(fig, directory, filename)

    return fig


##
# Show 2D images
# \date       2017-02-18 05:55:08+0000
#
# \param      images            The images
# \param      titles            The title
# \param      cmap              The cmap ("Greys_r", "jet", ...)
# \param      colorbar          The color
# \param      fontfamily        The fontfamily ("serif", "sans-serif", "monospace")
# \param      fontname          The fontname ("Arial", "Times New Roman")
# \param      use_tex           The use tex
# \param      fontsize          The fontsize
# \param      backgroundcolor   The backgroundcolor
# \param      aspect_ratio      The aspect ratio ("auto", "equal")
# \param      save_figure       The save fig
# \param      directory         The directory
# \param      filename          The filename including extension
# \param      fig_number        The fig number
# \param      use_same_scaling  The use same scaling
# \param      show_compact      The show compact
# \param      subplots_wspace   The subplots wspace
# \param      subplots_hspace   The subplots hspace
# \param      subplots_left     The subplots left
# \param      subplots_right    The subplots right
# \param      subplots_bottom   The subplots bottom
# \param      subplots_top      The subplots top
#
# \return     figure
#
def show_images(images, titles=None, cmap="Greys_r", use_colorbar=False, fontfamily="serif", fontname="Arial", use_tex=False, fontsize=12, backgroundcolor="None", aspect_ratio="auto", save_figure=False, directory=dir_tmp, filename="figure.pdf", fig_number=None, use_same_scaling=False, show_compact=False, subplots_wspace=0, subplots_hspace=0, subplots_left=0, subplots_right=1, subplots_bottom=0, subplots_top=0.85, grid_shape=None, figuresize=None):

    images = list(images)
    N_images = len(images)

    ## Change font
    font = {
        "family"        :   fontfamily,
        fontfamily      :   fontname,
        "size"          :   fontsize,
        }
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=use_tex)

    ## Define the grid to arrange the slices
    if grid_shape is None:
        grid = _get_grid_size(N_images)
    else:
        grid = grid_shape

    ## Use same scaling for all images
    if use_same_scaling:
        ## Extract min and max value of arrays for same scaling
        value_min = np.min(images[0])
        value_max = np.max(images[0])
        for i in range(1, N_images):
            value_min = np.min([value_min, np.min(images[i])])
            value_max = np.max([value_max, np.max(images[i])])
    else:
        value_min = None
        value_max = None


    # print("value_min = %.2f" %(value_min))
    # print("value_max = %.2f" %(value_max))
    ## Plot figure
    if show_compact:
        if figuresize is None:
            fig = plt.figure(fig_number, figsize=2*np.array(grid[::-1]))
        else:
            fig = plt.figure(fig_number, figsize=figuresize[::-1])

        plt.subplots_adjust(wspace=subplots_wspace, hspace=subplots_hspace, left=subplots_left, right=subplots_right, bottom=subplots_bottom, top=subplots_top)
    else:
        if figuresize is None:
            fig = plt.figure(fig_number)
        else:
            fig = plt.figure(fig_number, figsize=figuresize[::-1])

    fig.clf()


    for i in range(0, N_images):
        if i is 0:
            ax1 = plt.subplot(grid[0], grid[1], i+1)
            ax1.set_aspect(aspect_ratio)
        else:
            # ax2 = plt.subplot(gs1[i])
            ax2 = plt.subplot(grid[0], grid[1], i+1, sharex=ax1)
            ax2.set_aspect(aspect_ratio)
        im = plt.imshow(images[i], cmap=cmap, vmin=value_min, vmax=value_max)
        if titles is not None:
            if i < len(titles):
                plt.title(titles[i])
        
        ## Suppress axes
        plt.axis('off')
        
        ## Add colorbar on the side
        if use_colorbar:
            # add_axes([left, bottom, width, height])
            cax = fig.add_axes([0.92, 0.05, 0.01, 0.9])
            fig.colorbar(im, cax=cax)


    plt.show(block=False)

    if save_figure:
        
        ## Create directory in case it does not exist already
        create_directory(directory)

        ## Save figure to directory
        _save_figure(fig, directory, filename)

    return fig


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
def show_arrays(nda, title=None, cmap="Greys_r", use_colorbar=False, directory=None, fig_number=None, use_same_scaling=False):

    ## Show list of 2D arrays slice by slice
    if type(nda) is list:
        fig = _show_2D_array_list_array_by_array(nda, title=title, cmap=cmap, colorbar=use_colorbar, fig_number=fig_number, directory=directory, use_same_scaling=use_same_scaling)

    ## Show single 2D/3D array
    else:
        fig = show_array(nda, title=title, cmap=cmap, colorbar=use_colorbar, directory=directory, fig_number=fig_number)

    return fig

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
        fig = _show_2D_array(nda, title=title, cmap=cmap, colorbar=colorbar, directory=directory, save_type=save_type, fig_number=fig_number)

    ## Show single 3D array
    elif len(nda.shape) is 3:
        fig = _show_3D_array_slice_by_slice(nda, title=title, cmap=cmap, colorbar=colorbar, directory=directory, save_type=save_type, fig_number=fig_number)

    return fig


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

    return fig


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

    return fig


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
def _show_2D_array_list_array_by_array(nda2D_list, title=None, cmap="Greys_r", colorbar=False, fig_number=None, directory=None, axis_aspect='equal', use_same_scaling=False):

    # my_fontname = "Arial"
    my_fontname = "Times New Roman"
    title_font = {
        'fontname':my_fontname, 
        'size':'8', 
        'color':'black', 
        'weight':'normal',
        'verticalalignment':'center'  # Bottom vertical alignment for more space
        }


    shape = nda2D_list[0].shape
    N_slices = len(nda2D_list)

    if type(title) is not list and title is not None:
        title = [title]

    ## Define the grid to arrange the slices
    grid = _get_grid_size(N_slices)

    if use_same_scaling:
        ## Extract min and max value of arrays for same scaling
        value_min = np.min(nda2D_list[0])
        value_max = np.max(nda2D_list[0])
        for i in range(1, N_slices):
            value_min = np.min([value_min, np.min(nda2D_list[i])])
            value_max = np.max([value_max, np.max(nda2D_list[i])])
    else:
        value_min = None
        value_max = None


    # print("value_min = %.2f" %(value_min))
    # print("value_max = %.2f" %(value_max))
    ## Plot figure
    fig = plt.figure(fig_number, figsize=2*np.array(grid[::-1]))
    # fig = plt.figure(fig_number)
    fig.clf()


    # gs1 = gridspec.GridSpec(grid[0], grid[1])
    # gs1.update(wspace=0, hspace=0, ) #set spacing between axes
    
    for i in range(0, N_slices):
        if i is 0:
            # ax1 = plt.subplot(gs1[i])
            ax1 = plt.subplot(grid[0], grid[1], i+1)
            ax1.set_aspect(axis_aspect)
        else:
            # ax2 = plt.subplot(gs1[i])
            ax2 = plt.subplot(grid[0], grid[1], i+1, sharex=ax1)
            ax2.set_aspect(axis_aspect)
        im = plt.imshow(nda2D_list[i], cmap=cmap, vmin=value_min, vmax=value_max)
        if title is not None:
            if i < len(title):
                plt.title(title[i], **title_font)
        plt.axis('off')
        if colorbar:
            # add_axes([left, bottom, width, height])
            cax = fig.add_axes([0.92, 0.05, 0.01, 0.9])
            fig.colorbar(im, cax=cax)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=None)


    print("Slices of data arrays are shown in separate window.")
    plt.show(block=False)

    ## If directory is given: Save
    if directory is not None:
        ## Create directory in case it does not exist already
        create_directory(directory)

        filename = title[0]
        for i in range(1, N_slices):
            if i < len(title):
                filename += "_" + title[i]


        ## Save figure to directory
        _save_figure(fig, directory, filename)

    return fig

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
    if N_slices <= 4:
        grid = (1, N_slices)
    elif N_slices > 4 and N_slices <= 10:
        grid = (2, np.ceil(N_slices/2.).astype('int'))
    elif N_slices > 10 and N_slices <= 15:
        grid = (3, np.ceil(N_slices/3.).astype('int'))
    elif N_slices > 15 and N_slices <= 20:
        grid = (4, np.ceil(N_slices/3.).astype('int'))
    elif N_slices > 21 and N_slices <= 30:
        grid = (5, np.ceil(N_slices/4.).astype('int'))
    else:
        grid = (6, np.ceil(N_slices/5.).astype('int'))

    return grid


##
# Saves a figure to given directory
# \date       2017-02-07 10:19:09+0000
#
# \param      fig        The fig
# \param      directory  The directory
# \param      filename   The filename including filename extension
#
def _save_figure(fig, directory, filename):

    ## Add backslash if not given
    if directory[-1] is not "/":
        directory += "/"

    fig.savefig(directory + filename)
    print("Figure was saved to " + directory + filename)


##
# Closes all pyplot figures.
# \date       2017-02-07 10:30:57+0000
#
def close_all_figures():
    plt.close('all')


##
# Returns start time of execution
# \date       2016-11-06 17:15:00+0000
#
# \return     Start time of execution
#
def start_timing():
    return time.time()


##
# Adds times obtained by start and stop timing
# \date       2017-06-08 15:38:27+0100
#
# \param      time_1  time obtained by stop_timing
# \param      time_2  The time 2
#
# \return     added time
#
def add_times(time_1, time_2):
    return time_1 + time_2


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


def print_debug_info(text, newline=True, prefix="--- "):
    
    # print("---- Debug info: ----")
    if newline:
        print(prefix + text)
    else:
        sys.stdout.write(prefix + text)
    # print("---------------------------")


def print_title(text, symbol="*", add_newline=False):
    print_line_separator(symbol=symbol)
    print_subtitle(text,symbol=symbol, add_newline=add_newline)


def print_subtitle(text, symbol="*", add_newline=True):
    if add_newline:
        print("")
    print(3*symbol + " " + text + " " + 3*symbol)


def print_line_separator(add_newline=True, symbol="*", length=99):
    if add_newline:
        print("\n")
    print(symbol*length)


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
        directory += "/"

    ## Create directory in case it does not exist already
    if not os.path.isdir(directory):
        os.system("mkdir -p " + directory)
        print_debug_info("Directory " + directory + " created.")

    if delete_files:
        clear_directory(directory)

    return directory


##
# Clear all data in given directory
# \date       2017-02-02 16:47:15+0000
#
# \param      directory  The directory
#
def clear_directory(directory):

    if directory[-1] not in ["/"]:
        directory += "/"

    os.system("rm -rf " + directory + "*")
    print_debug_info("All files in " + directory + " are removed.")

def delete_directory(directory):
    os.system("rm -rf " + directory)
    print_debug_info("Directory " + directory + " deleted.")


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
def create_image_grid(shape, spacing, value_bg=1, value_fg=0):

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
def create_image_with_slope(shape, slope=1, value_bg=0, value_fg=1, offset=0):

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
def create_image_pyramid(length, slope=1, value_bg=0, value_fg=1, offset=(0,0)):

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
    nda = np.round(np.array(nda))
    im = Image.fromarray(nda)
    im.save(filename)
