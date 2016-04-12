## \file Test_Stack.py
#  \brief  Class containing unit tests for module Stack
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


## Import libraries 
import SimpleITK as sitk
import numpy as np
import unittest


## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class Stack(unittest.TestCase):
    accuracy = 2

    def setUp(self):
        pass

    def test_01_transformation_back_to_origin(self):
        self.assertEqual(1, self.accuracy)
