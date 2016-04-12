import SimpleITK as sitk
import numpy as np
import unittest
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.stats import chi2

import sys
sys.path.append("../src")

import SimpleITKHelper as sitkh

"""
Functions
"""

"""
Unit Test Class
"""
class TestUM(unittest.TestCase):

    accuracy = 8

    # def compute_single_value(self, point, origin, Sigma):
    #     return (point-origin).dot(np.linalg.inv(Sigma)).dot(point-origin)


    def setUp(self):
        pass

    def test_01(self):
        pass


"""
Main
"""
## Specify data
dir_input = "data/"
dir_output = "results/"

filename =  "placenta_s"
filename = "BrainWeb_2D"
# filename =  "kidney_s"
# filename =  "fetal_brain_a"
# filename =  "fetal_brain_c"
# filename =  "fetal_brain_s"

image_type = ".png"
# image_type = ".nii.gz"

## Read image
# image_sitk = sitk.ReadImage(dir_input + filename + image_type) 

##

data = np.zeros((5,5))
data[2,2] = 1

# kernel = np.arange(1,10).reshape(3,-1)
# kernel = np.arange(1,5).reshape(2,-1)
kernel = np.arange(1,7).reshape(2,-1)
# kernel = np.arange(1,7).reshape(3,-1)

print("data = \n%s" %(data))
print("kernel = \n%s\n" %(kernel))

data_convolved = ndimage.convolve(data, kernel, mode='constant')
print("data convolved without stated origin\n%s" %(data_convolved))

origin = 0 # default value
# data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
# print("data convolved with origin=%s\n%s" %(origin, data_convolved))

# origin = (0,0) # same as default value
# data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
# print("data convolved with origin=%s\n%s" %(origin, data_convolved))

# origin = (1,0)
data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
print("data convolved with origin=%s\n%s" %(origin, data_convolved))

origin = (-1,0)
data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
print("data convolved with origin=%s\n%s" %(origin, data_convolved))

origin = (0,1)
data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
print("data convolved with origin=%s\n%s" %(origin, data_convolved))

origin = (0,-1)
data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
print("data convolved with origin=%s\n%s" %(origin, data_convolved))

# origin = (1,1)
data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
print("data convolved with origin=%s\n%s" %(origin, data_convolved))

# origin = 1    # same as (1,1)
data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
print("data convolved with origin=%s\n%s" %(origin, data_convolved))

origin = (-1,-1)
data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
print("data convolved with origin=%s\n%s" %(origin, data_convolved))

origin = -1     # same as (-1,-1)
data_convolved = ndimage.convolve(data, kernel, mode='constant', origin=origin)
print("data convolved with origin=%s\n%s" %(origin, data_convolved))

print("\nkernel = \n%s" %(kernel))
print("standard coordinates for origin within array are given by np.array(kernel.shape)/2 = \n%s" %(np.array(kernel.shape)/2))

"""
Unit tests:
"""
# print("\nUnit tests:\n--------------")
# unittest.main()