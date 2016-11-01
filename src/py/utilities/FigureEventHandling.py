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

##-----------------------------------------------------------------------------
# \brief      Class used to semi-automatically extract position and geometry of
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

    ##-------------------------------------------------------------------------
    # \brief      Constructor
    # \date       2016-09-18 02:45:09+0100
    #
    # \param      self  The object
    # \param      nda   2D Numpy data array representing the film with all
    #                   acquired 2D slices
    #
    def __init__(self, nda):
        self._nda = nda
        self._coordinates = []
        self._rectangles = []

        ## Tested for subject 2
        ## Parameters in case left white circle exists (Click center of left circle)
        self._offset = np.array([100,-900])
        self._length = np.array([1350,1600])

        ## Parameters in case left white circle do not exist (Click L-corner in "LEFT")
        # self._offset = np.array([-1450,-550])
        # self._length = np.array([1100,1250])

        ## Used for navigation through x-offset, y-offset, x-length and y-length
        self._index = 0


    ##-------------------------------------------------------------------------
    # \brief      Gets the marked coordinates.
    # \date       2016-09-16 16:16:47+0100
    #
    # \param      self  The object
    #
    # \return     The coordinates.
    #
    def get_coordinates(self):
        return self._coordinates


    ##-------------------------------------------------------------------------
    # \brief      Sets the offset of the region with respect to set coordinates
    #             on image.
    # \date       2016-09-19 13:35:57+0100
    #
    # \param      self    The object
    # \param      offset  The offset as 2D array
    #
    def set_offset(self, offset):
        self._offset = offset


    ##-------------------------------------------------------------------------
    # \brief      Gets the offset.
    # \date       2016-09-17 22:29:33+0100
    #
    # \param      self  The object
    #
    # \return     The offset.
    #
    def get_offset(self):
        return self._offset


    ##-------------------------------------------------------------------------
    # \brief      Sets the length of the region with respect to the set
    #             coordinates on image
    # \date       2016-09-19 13:36:50+0100
    #
    # \param      self    The object
    # \param      length  The length as 2D array
    #
    def set_length(self, length):
        self._length = length


    ##-------------------------------------------------------------------------
    # \brief      Gets the length.
    # \date       2016-09-17 22:30:10+0100
    #
    # \param      self  The object
    #
    # \return     The length.
    #
    def get_length(self):
        return self._length


    ##-------------------------------------------------------------------------
    # \brief      Plot 2D array and extract slices by clicking on the figure.
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
        plt.show()


    ##-------------------------------------------------------------------------
    # \brief      Print information of use on screen
    # \date       2016-09-18 01:33:20+0100
    #
    # \param      self  The object
    #
    def _print_help(self):
        print("Use the following keys to navigate: ")
        print("\tmiddle click: Click on point to save coordinates.")
        print("\td:      Delete most recent coordinates.")
        print("\tright:  Move up to switch between x-offset, y-offset, x-length and y-length.")
        print("\tleft:   Move down to switch between x-offset, y-offset, x-length and y-length.")
        print("\tup:     Increase chosen property by one.")
        print("\tdown:   Decrease chosen property by one.")
        print("\tspace:  Use keyboard to define value of chosen property.")
        print("\tescape: Close figure. Values for coordinates, offset and length are stored.")
        print("\th:      Print this information.")
        print("\tp:      Switch type of mouse (select and move/zoom).")


    ##-------------------------------------------------------------------------
    # \brief      Event handling for image clicks on plots. Used to store
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
            print("%d.Point (%d, %d)" %(len(self._coordinates)+1, event.xdata, event.ydata))
            
            ## Get rounded integer value
            xdata = np.round(event.xdata).astype(int)
            ydata = np.round(event.ydata).astype(int)

            ## Save coordinates of selected point
            self._coordinates.append([xdata, ydata])

            ## Redraw in order to draw selected point on image
            self._redraw()
            
        # elif event.button is 1:
            # print("Left click")
        # elif event.button is 3:
            # print("Right click")


    ##-------------------------------------------------------------------------
    # \brief      Event handling for hitting keys
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
            self._redraw()

        ## Increase respective value by one
        if event.key in ["up"]:
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
        if event.key in ["down"]:
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

        ## Use console for value input of respective option
        if event.key in [" "]:
            if self._index is 0:
                self._offset[0] = input("Enter value of x-offset [%s]: " %(self._offset[0]))
            if self._index is 1:
                self._offset[1] = input("Enter value of y-offset [%s]: " %(self._offset[1]))
            if self._index is 2:
                self._length[0] = input("Enter value of x-length [%s]: " %(self._length[0]))
            if self._index is 3:
                self._length[1] = input("Enter value of y-length [%s]: " %(self._length[1]))

            ## Delete all rectangles before updated ones are drawn
            self._delete_previous_rectangles()
            self._redraw()

        ## Select option to the right in sequence x-offset, y-offset, x-length, y-length
        if event.key in ["right"]:
            if self._index<3:
                self._index += 1

            if self._index is 0:
                print("Chosen option: x-offset. Use either 'space' or arrows 'up' and 'down' to adjust its value.")
            if self._index is 1:
                print("Chosen option: y-offset. Use either 'space' or arrows 'up' and 'down' to adjust its value.")
            if self._index is 2:
                print("Chosen option: x-length. Use either 'space' or arrows 'up' and 'down' to adjust its value.")
            if self._index is 3:
                print("Chosen option: y-length. Use either 'space' or arrows 'up' and 'down' to adjust its value.")

        ## Select option to the left in sequence x-offset, y-offset, x-length, y-length
        if event.key in ["left"]:
            if self._index>0:
                self._index -= 1

            if self._index is 0:
                print("Chosen option: x-offset. Use either 'space' or arrows 'up' and 'down' to adjust its value.")
            if self._index is 1:
                print("Chosen option: y-offset. Use either 'space' or arrows 'up' and 'down' to adjust its value.")
            if self._index is 2:
                print("Chosen option: x-length. Use either 'space' or arrows 'up' and 'down' to adjust its value.")
            if self._index is 3:
                print("Chosen option: y-length. Use either 'space' or arrows 'up' and 'down' to adjust its value.")

        ## Print help information
        if event.key in ["h"]:
            self._print_help()

        ## Close
        if event.key in ["escape"]:
            print("Close window ...")
            plt.close()


    ##-------------------------------------------------------------------------
    # \brief      Redraw points and rectangle in case any update has happened
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


    ##-------------------------------------------------------------------------
    # \brief      Delete all drawn rectangles after update has happened
    # \date       2016-09-18 02:42:00+0100
    #
    # \param      self  The object
    #
    def _delete_previous_rectangles(self):
        # self._rectangles[-1].remove()
        for i in range(0,len(self._rectangles)):
            self._rectangles[i].set_visible(False)
            # self._rectangles[i].remove()

