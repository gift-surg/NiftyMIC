# \file TestDifferentialOperations.py
#  \brief  Class containing unit tests for module DifferentialOperations
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date July 2016


# Import libraries
import SimpleITK as sitk
import numpy as np
import unittest
import sys

# Import modules from src-folder
import niftymic.base.stack as st
import niftymic.reconstruction.solver.differential_operations as diffop

from niftymic.definitions import DIR_TEST


class DifferentialOperationsTest(unittest.TestCase):

    # Specify input data
    dir_test_data = DIR_TEST

    accuracy = 6

    def setUp(self):
        pass

    # Test whether |(Dx,y) - (x,D'y)| for D being the differential operator
    #  for each single direction. Periodic boundary conditions are chosen
    def test_adjoint_differentiation_convolution_mode_wrap(self):

        nda_shape = (50, 60, 70)
        nda_x = 255 * np.random.rand(nda_shape[0], nda_shape[1], nda_shape[2])
        nda_y = 255 * np.random.rand(nda_shape[0], nda_shape[1], nda_shape[2])

        differential_operations = diffop.DifferentialOperations(
            step_size=0.5, convolution_mode='wrap')

        # Check |(Dx,y) - (x,D'y)| = 0 for differentiation in x
        Dx = differential_operations.Dx(nda_x)
        DTy = differential_operations.Dx_adj(nda_y)
        abs_diff = np.abs(np.sum(Dx*nda_y) - np.sum(DTy*nda_x))
        # print("Periodic boundary conditions (wrap)")
        # print("\tx-derivative: |(Dx,y) - (x,D'y)| = %s" %(abs_diff))

        self.assertEqual(np.around(
            abs_diff, decimals=self.accuracy), 0)

        # Check |(Dx,y) - (x,D'y)| = 0 for differentiation in y
        Dx = differential_operations.Dy(nda_x)
        DTy = differential_operations.Dy_adj(nda_y)
        abs_diff = np.abs(np.sum(Dx*nda_y) - np.sum(DTy*nda_x))
        # print("\ty-derivative: |(Dx,y) - (x,D'y)| = %s" %(abs_diff))

        self.assertEqual(np.around(
            abs_diff, decimals=self.accuracy), 0)

        # Check |(Dx,y) - (x,D'y)| = 0 for differentiation in z
        Dx = differential_operations.Dz(nda_x)
        DTy = differential_operations.Dz_adj(nda_y)
        abs_diff = np.abs(np.sum(Dx*nda_y) - np.sum(DTy*nda_x))
        # print("\tz-derivative: |(Dx,y) - (x,D'y)| = %s" %(abs_diff))

        self.assertEqual(np.around(
            abs_diff, decimals=self.accuracy), 0)

    # Test whether |(Dx,y) - (x,D'y)| for D being the differential operator
    #  for each single direction. Zero boundary conditions are chosen
    def test_adjoint_differentiation_convolution_mode_constant(self):

        nda_shape = (50, 60, 70)
        nda_x = 255 * np.random.rand(nda_shape[0], nda_shape[1], nda_shape[2])
        nda_y = 255 * np.random.rand(nda_shape[0], nda_shape[1], nda_shape[2])

        differential_operations = diffop.DifferentialOperations(
            step_size=0.5, convolution_mode='constant')

        # Check |(Dx,y) - (x,D'y)| = 0 for differentiation in x
        Dx = differential_operations.Dx(nda_x)
        DTy = differential_operations.Dx_adj(nda_y)
        abs_diff = np.abs(np.sum(Dx*nda_y) - np.sum(DTy*nda_x))
        # print("Zero boundary conditions (constant)")
        # print("\tx-derivative: |(Dx,y) - (x,D'y)| = %s" %(abs_diff))

        self.assertEqual(np.around(
            abs_diff, decimals=self.accuracy), 0)

        # Check |(Dx,y) - (x,D'y)| = 0 for differentiation in y
        Dx = differential_operations.Dy(nda_x)
        DTy = differential_operations.Dy_adj(nda_y)
        abs_diff = np.abs(np.sum(Dx*nda_y) - np.sum(DTy*nda_x))
        # print("\ty-derivative: |(Dx,y) - (x,D'y)| = %s" %(abs_diff))

        self.assertEqual(np.around(
            abs_diff, decimals=self.accuracy), 0)

        # Check |(Dx,y) - (x,D'y)| = 0 for differentiation in z
        Dx = differential_operations.Dz(nda_x)
        DTy = differential_operations.Dz_adj(nda_y)
        abs_diff = np.abs(np.sum(Dx*nda_y) - np.sum(DTy*nda_x))
        # print("\tz-derivative: |(Dx,y) - (x,D'y)| = %s" %(abs_diff))

        self.assertEqual(np.around(
            abs_diff, decimals=self.accuracy), 0)

    # Check whether Laplace f = (Dx + Dx' + Dy + Dy' + Dz + Dz') f for step_size=1
    # for the Laplace stencil computation. Periodic boundary conditions are
    # chosen
    def test_Laplace_stencil_convolution_mode_wrap(self):

        nda_shape = (50, 60, 70)
        nda_x = 255 * np.random.rand(nda_shape[0], nda_shape[1], nda_shape[2])
        nda_y = 255 * np.random.rand(nda_shape[0], nda_shape[1], nda_shape[2])

        differential_operations = diffop.DifferentialOperations(
            step_size=1, Laplace_comp_type="LaplaceStencil", convolution_mode='wrap')

        # 1) Test Laplace f = (Dx + Dx' + Dy + Dy' + Dz + Dz') f for
        # step_size=1
        Lx = differential_operations.Laplace(nda_x)
        Dx = differential_operations.Dx(nda_x)
        Dx_adj = differential_operations.Dx_adj(nda_x)
        Dy = differential_operations.Dy(nda_x)
        Dy_adj = differential_operations.Dy_adj(nda_x)
        Dz = differential_operations.Dz(nda_x)
        Dz_adj = differential_operations.Dz_adj(nda_x)
        norm_diff = np.linalg.norm(
            Lx - (Dx + Dx_adj + Dy + Dy_adj + Dz + Dz_adj))

        self.assertEqual(np.around(
            norm_diff, decimals=self.accuracy), 0)

        # 2) Test Laplace is self-adjoint, i.e. |(Lx,y) - (x,Ly)| = 0
        Ly = differential_operations.Laplace(nda_y)
        abs_diff = np.abs(np.sum(Lx*nda_y) - np.sum(Ly*nda_x))

        self.assertEqual(np.around(
            abs_diff, decimals=self.accuracy), 0)

    # Check whether Laplace f = (Dx + Dx' + Dy + Dy' + Dz + Dz') f for step_size=1
    # for the Laplace stencil computation. Periodic boundary conditions are
    # chosen
    def test_Laplace_stencil_convolution_mode_constant(self):

        nda_shape = (50, 60, 70)
        nda_x = 255 * np.random.rand(nda_shape[0], nda_shape[1], nda_shape[2])
        nda_y = 255 * np.random.rand(nda_shape[0], nda_shape[1], nda_shape[2])

        differential_operations = diffop.DifferentialOperations(
            step_size=1, Laplace_comp_type="LaplaceStencil", convolution_mode='constant')

        # 1) Test Laplace f = (Dx + Dx' + Dy + Dy' + Dz + Dz') f for
        # step_size=1
        Lx = differential_operations.Laplace(nda_x)
        Dx = differential_operations.Dx(nda_x)
        Dx_adj = differential_operations.Dx_adj(nda_x)
        Dy = differential_operations.Dy(nda_x)
        Dy_adj = differential_operations.Dy_adj(nda_x)
        Dz = differential_operations.Dz(nda_x)
        Dz_adj = differential_operations.Dz_adj(nda_x)
        norm_diff = np.linalg.norm(
            Lx - (Dx + Dx_adj + Dy + Dy_adj + Dz + Dz_adj))

        self.assertEqual(np.around(
            norm_diff, decimals=self.accuracy), 0)

        # 2) Test Laplace is self-adjoint, i.e. |(Lx,y) - (x,Ly)| = 0
        Ly = differential_operations.Laplace(nda_y)
        abs_diff = np.abs(np.sum(Lx*nda_y) - np.sum(Ly*nda_x))

        self.assertEqual(np.around(
            abs_diff, decimals=self.accuracy), 0)
