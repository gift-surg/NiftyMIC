#!/usr/bin/python

## \file FigureEventHandling.py
#  \brief 
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Sept 2016


## Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

## Import modules
import pythonhelper.PythonHelper as ph

##
#       Class used to semi-automatically extract position and geometry of
#             slices
# \date       2016-09-16 16:05:06+0100
#
# Input consists of several 2D scans printed on one film. This class serves to
# extract all slices as single acquisition in an semi-automatic fashion. By
# clicking on the image the coordinates of this position is saved. Together
# with the offset and length of the rectangle (can be adjusted) the window
# containing the scan can be extracted. In further processing these can be
# stacked to one 3D stack of slices.
#
class FigureEventHandling(object):

    #--------------------------------------------------------------------------
    # \brief      Constructor
    # \date       2016-09-18 02:45:09+0100
    #
    # \param      self   The object
    # \param      nda    2D Numpy data array representing the film with all
    #                    acquired 2D slices
    # \param      title  The title used for the data array plot in matploblib
    #
    def __init__(self, nda, title=None):
        self._nda = nda
        self._coordinates = []
        self._rectangles = []

        ## Tested for subject 2
        ## Parameters in case left white circle exists (Click center of left circle)
        # self._offset = np.array([100,-900])
        # self._length = np.array([1350,1600])

        ## Parameters in case left white circle do not exist (Click L-corner in "LEFT")
        # self._offset = np.array([-1450,-550])
        # self._length = np.array([1100,1300])

        ## Used for navigation through x-offset, y-offset, x-length and y-length
        self._index = 0

        ## Title used for the data array plot
        self._title = title


        ## Single and double mode
        self._double_mode_dict = {
            False:      "single",
            True:       "double"
        }
        self._double_mode = False

        ## Define bookmarks for quicker switching between selection frames        
        self._bookmark_offset = {
              "left_circle"           : np.array([100,-900])
            , "bottom_left_corner"    : np.array([600,-1650])
            , "top_left_corner"       : np.array([600,150])
            , "bottom_right_corner"   : np.array([-1800,-1850])
            , "L_corner"              : np.array([-1450,-550])
            , "double_window_5yr"     : np.array([10,-10])
            , "double_window_10yr"    : np.array([8,15])
        }
        self._bookmark_length = {
              "left_circle"           : np.array([1350,1700])
            , "bottom_left_corner"    : np.array([1150,1500])
            , "top_left_corner"       : np.array([1300,1700])
            , "bottom_right_corner"   : np.array([1300,1700])
            , "L_corner"              : np.array([1100,1300])
            , "double_window_5yr"     : np.array([853,1400])
            , "double_window_10yr"    : np.array([685,1024])
        }

        ## Set offset for bookmarks
        self._bookmark_default_integer = 3

        keys = self._bookmark_length.keys()
        self._offset = self._bookmark_offset[keys[self._bookmark_default_integer]]
        self._length = self._bookmark_length[keys[self._bookmark_default_integer]]

    ##
    #       Gets the marked coordinates.
    # \date       2016-09-16 16:16:47+0100
    #
    # \param      self  The object
    #
    # \return     The coordinates.
    #
    def get_coordinates(self):
        return self._coordinates


    ##
    #       Sets the offset of the region with respect to set coordinates
    #             on image.
    # \date       2016-09-19 13:35:57+0100
    #
    # \param      self    The object
    # \param      offset  The offset as 2D array
    #
    def set_offset(self, offset):
        self._offset = offset


    def set_bookmark_default_integer(self, bookmark_default_integer):
        self._bookmark_default_integer = bookmark_default_integer

        keys = self._bookmark_length.keys()
        self._offset = self._bookmark_offset[keys[self._bookmark_default_integer]]
        self._length = self._bookmark_length[keys[self._bookmark_default_integer]]


    ##
    #       Gets the offset.
    # \date       2016-09-17 22:29:33+0100
    #
    # \param      self  The object
    #
    # \return     The offset.
    #
    def get_offset(self):
        return self._offset


    ##
    #       Sets the length of the region with respect to the set
    #             coordinates on image
    # \date       2016-09-19 13:36:50+0100
    #
    # \param      self    The object
    # \param      length  The length as 2D array
    #
    def set_length(self, length):
        self._length = length


    ##
    #       Gets the length.
    # \date       2016-09-17 22:30:10+0100
    #
    # \param      self  The object
    #
    # \return     The length.
    #
    def get_length(self):
        return self._length


    ##
    #       Plot 2D array and extract slices by clicking on the figure.
    #             Exit this process by hitting enter.
    # \date       2016-09-16 15:50:09+0100
    #
    # \param      self  The object
    #
    def extract_slices_semiautomatically(self):
        fig_number = 1

        ## Create plot
        self._fig = plt.figure(fig_number)
        self._ax = self._fig.add_subplot(1,1,1)
        self._canvas = self._ax.get_figure().canvas

        ## Add possibility to add coordinates after each click and show them on 
        ## image
        self._pt_plot = self._ax.plot([], [], 'bx', markersize=5)[0]

        ## Add event handling
        cid = self._fig.canvas.mpl_connect('button_press_event', self._event_onclick)
        cid = self._fig.canvas.mpl_connect('key_press_event', self._event_onkey)

        ## Print help information on screen
        self._print_help()

        ## Plot plain image in gray scale. aspect=auto yields more convenient
        ## view to work on in fullscreen
        self._ax.imshow(self._nda, cmap="Greys_r", aspect='auto')

        if self._title is not None:
            plt.title(self._title)

        plt.show()


    ##
    #       Print information of usage on screen
    # \date       2016-09-18 01:33:20+0100
    #
    # \param      self  The object
    #
    def _print_help(self):

        keys = self._bookmark_length.keys()
            # print("Bookmark '" + keys[bookmark] + "' is chosen." )

        ph.print_line_separator(symbol="-")
        ph.print_subtitle("Help: List of Navigation Keys",symbol="-")
        
        print("\nGeneral Handling")
        print("\th:            Print this information.")
        print("\tp:            Switch mouse cursor type (select and move/zoom).")
        print("\t              'Cross':   Zoom in/out with held right click.")
        print("\t                         (Hit 'r' to zoom out entirely again).")
        print("\t              'Arrow':   More precise selection possible.")
        print("\tMiddle click: Click on image position to save its coordinates.")
        print("\td:            Delete most recent point coordinates.")
        print("\tEsc:          Close figure and continue with next MR film (in case existing).")
        print("\t              Selected coordinates and cropping window (offset and length) are stored.")

        print("\nAdapt Selection Window:")
        print("\tb:            Choose among %s bookmarks to define selection box dimension [%s]." %(len(self._bookmark_length.keys()), keys[self._bookmark_default_integer]))
        print("\tright:        Move up to switch between x-offset, y-offset, x-length and y-length.")
        print("\tleft:         Move down to switch between x-offset, y-offset, x-length and y-length.")
        print("\t              up:        Increase chosen property by one pixel.")
        print("\t              down:      Decrease chosen property by one pixel.")
        print("\t              pageup:    Increase chosen property by 50 pixels.")
        print("\t              pagedown:  Decrease chosen property by 50 pixels.")
        print("\t              space:     Use keyboard to define value of chosen property.")

        print("\nSingle/Double Mode:")
        print("\tm:            Change between single and double selection mode [%s]." %(self._double_mode_dict[self._double_mode]))
        
        ph.print_line_separator(add_newline=False,symbol="-")

    ##
    #       Event handling for image clicks on plots. Used to store
    #             coordinates of clicked position
    # \see        http://matplotlib.org/users/event_handling.html
    # \date       2016-09-16 15:51:00+0100
    #
    # \param      self   The object
    # \param      event  The event
    #
    def _event_onclick(self, event):

        ## Only store in case of middle click
        if event.button is 2:
            # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              # (event.button, event.x, event.y, event.xdata, event.ydata))
            
            ## Get rounded integer value
            xdata = np.round(event.xdata).astype(int)
            ydata = np.round(event.ydata).astype(int)

            if self._double_mode:
                xdatapoint = xdata-2*self._offset[0]-self._length[0]
                ydatapoint = ydata
                print("%d.Point (%d, %d)" %(len(self._coordinates)+1, xdatapoint, ydatapoint))
                self._coordinates.append([xdatapoint, ydatapoint])

            ## Save coordinates of selected point
            xdatapoint = xdata
            ydatapoint = ydata
            print("%d.Point (%d, %d)" %(len(self._coordinates)+1, xdatapoint, ydatapoint))
            self._coordinates.append([xdata, ydata])

            ## Redraw in order to draw selected point on image
            self._redraw()
            
        # elif event.button is 1:
            # print("Left click")
        # elif event.button is 3:
            # print("Right click")


    ##
    #       Event handling for hitting keys
    # \see        http://matplotlib.org/users/event_handling.html
    # \date       2016-09-16 15:51:53+0100
    #
    # \param      self   The object
    # \param      event  The event
    #
    def _event_onkey(self, event):
        
        # print('key=%s' % (event.key))

        ## Delete last coordinates
        if event.key in ["d"]:
            if len(self._coordinates)>0:
                foo = self._coordinates.pop()
                print("Deleted last entry: %s" %(foo))
                ## Delete all rectangles before updated ones are drawn
                self._delete_previous_rectangles()
            self._redraw()

        if event.key in ["m"]:
            if self._double_mode is False:
                self._double_mode = True
                print("Continue in double mode.")
            else:
                self._double_mode = False
                print("Continue in single mode.")

        ## Choose dimension + offset of sele
        elif event.key in ["b"]:
            ## Get the bookmark keys and put them together for the info text
            keys = self._bookmark_length.keys()
            text = "\n\t0: " + keys[0]
            for i in range(1, len(keys)):
                text += ",\n\t" + str(i) + ": " + keys[i]
            
            ## Read bookmark selection
            bookmark = int(ph.read_input("Chose number to select bookmark:" + text + "\n", default=self._bookmark_default_integer))
            
            ## Update selection box accordingly
            self._offset = self._bookmark_offset[keys[bookmark]]
            self._length = self._bookmark_length[keys[bookmark]]
            self._bookmark_default_integer = bookmark
            print("Bookmark '" + keys[bookmark] + "' is chosen." )
            
            ## Delete all rectangles before updated ones are drawn
            self._delete_previous_rectangles()
            self._redraw()

        ## Increase respective value by one
        elif event.key in ["up"]:
            if self._index is 0:
                self._offset[0] += 1
            if self._index is 1:
                self._offset[1] += 1
            if self._index is 2:
                self._length[0] += 1
            if self._index is 3:
                self._length[1] += 1

            ## Delete all rectangles before updated ones are drawn
            self._delete_previous_rectangles()
            self._redraw()

        ## Decrease respective value by one
        elif event.key in ["down"]:
            if self._index is 0:
                self._offset[0] -= 1
            if self._index is 1:
                self._offset[1] -= 1
            if self._index is 2:
                self._length[0] -= 1
            if self._index is 3:
                self._length[1] -= 1

            ## Delete all rectangles before updated ones are drawn
            self._delete_previous_rectangles()
            self._redraw()

        ## Increase respective value by 50
        elif event.key in ["pageup"]:
            if self._index is 0:
                self._offset[0] += 50
            if self._index is 1:
                self._offset[1] += 50
            if self._index is 2:
                self._length[0] += 50
            if self._index is 3:
                self._length[1] += 50

            ## Delete all rectangles before updated ones are drawn
            self._delete_previous_rectangles()
            self._redraw()

        ## Decrease respective value by 50
        elif event.key in ["pagedown"]:
            if self._index is 0:
                self._offset[0] -= 50
            if self._index is 1:
                self._offset[1] -= 50
            if self._index is 2:
                self._length[0] -= 50
            if self._index is 3:
                self._length[1] -= 50

            ## Delete all rectangles before updated ones are drawn
            self._delete_previous_rectangles()
            self._redraw()

        ## Use console for value input of respective option
        elif event.key in [" "]:
            if self._index is 0:
                self._offset[0] = int(float(ph.read_input("Enter value for x-offset", default=self._offset[0])))
            if self._index is 1:
                self._offset[1] = int(float(ph.read_input("Enter value for y-offset", default=self._offset[1])))
            if self._index is 2:
                self._length[0] = int(float(ph.read_input("Enter value for x-length", default=self._length[0])))
            if self._index is 3:
                self._length[1] = int(float(ph.read_input("Enter value for y-length", default=self._length[1])))

            ## Delete all rectangles before updated ones are drawn
            self._delete_previous_rectangles()
            self._redraw()

        ## Select option to the right in sequence x-offset, y-offset, x-length, y-length
        elif event.key in ["right"]:
            if self._index<3:
                self._index += 1

            if self._index is 0:
                print("Chosen option: x-offset. Use either 'space' or arrows 'up', 'down', 'pageup' and 'pagedown' to adjust its value.")
            if self._index is 1:
                print("Chosen option: y-offset. Use either 'space' or arrows 'up', 'down', 'pageup' and 'pagedown' to adjust its value.")
            if self._index is 2:
                print("Chosen option: x-length. Use either 'space' or arrows 'up', 'down', 'pageup' and 'pagedown' to adjust its value.")
            if self._index is 3:
                print("Chosen option: y-length. Use either 'space' or arrows 'up', 'down', 'pageup' and 'pagedown' to adjust its value.")

        ## Select option to the left in sequence x-offset, y-offset, x-length, y-length
        elif event.key in ["left"]:
            if self._index>0:
                self._index -= 1

            if self._index is 0:
                print("Chosen option: x-offset. Use either 'space' or arrows 'up', 'down', 'pageup' and 'pagedown' to adjust its value.")
            if self._index is 1:
                print("Chosen option: y-offset. Use either 'space' or arrows 'up', 'down', 'pageup' and 'pagedown' to adjust its value.")
            if self._index is 2:
                print("Chosen option: x-length. Use either 'space' or arrows 'up', 'down', 'pageup' and 'pagedown' to adjust its value.")
            if self._index is 3:
                print("Chosen option: y-length. Use either 'space' or arrows 'up', 'down', 'pageup' and 'pagedown' to adjust its value.")

        ## Print help information
        elif event.key in ["h"]:
            self._print_help()

        ## Close
        elif event.key in ["escape"]:
            print("Close window ...")
            plt.close()


    ##
    #       Redraw points and rectangle in case any update has happened
    # \date       2016-09-16 17:03:12+0100
    # \see        http://stackoverflow.com/questions/19592422/python-gui-that-draw-a-dot-when-clicking-on-plot
    #
    # \param      self  The object
    #
    def _redraw(self):
        N = len(self._coordinates)
        N_rectangles = len(self._rectangles)

        ## Plot points
        if N>0:
            x,y = zip(*self._coordinates)
            self._pt_plot.set_xdata(x)
            self._pt_plot.set_ydata(y)
        else:
            self._pt_plot.set_xdata([])
            self._pt_plot.set_ydata([])
        self._canvas.draw()

        ## Show current offset and length after every update
        print("offset=(%s, %s), length=(%s, %s)" %(self._offset[0], self._offset[1], self._length[0], self._length[1]))
     
        ## Plot rectangles
        for i in range(0, N):
            self._rectangles.append(Rectangle((x[i]+self._offset[0], y[i]+self._offset[1]), self._length[0], self._length[1], alpha=1, fill=None, edgecolor='r', linewidth=1))
            self._ax.add_patch(self._rectangles[-1])
        plt.draw()


    ##
    #       Delete all drawn rectangles after update has happened
    # \date       2016-09-18 02:42:00+0100
    #
    # \param      self  The object
    #
    def _delete_previous_rectangles(self):
        # self._rectangles[-1].remove()
        for i in range(0,len(self._rectangles)):
            self._rectangles[i].set_visible(False)
            # self._rectangles[i].remove()

