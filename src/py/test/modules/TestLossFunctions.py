##
# \file TestLossFunctions.py
#  \brief  Class containing unit tests for module Stack
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


# Import libraries
import numpy as np
import unittest
import sys
import matplotlib.pyplot as plt

# Import modules
import utilities.lossFunctions as lf
# import utilities.SimpleITKHelper as sitkh
import utilities.PythonHelper as ph

from definitions import dir_test

# Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015


class TestNiftyReg(unittest.TestCase):

    # Specify input data
    dir_test_data = dir_test

    accuracy = 7
    m = 500     # 4e5
    n = 1000    # 1e6

    def setUp(self):
        pass

    def test_linear(self):

        b = np.random.rand(self.m)

        diff = lf.linear(b) - b

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff = lf.gradient_linear(b) - 1
        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

    def test_least_squares_linear(self):

        A = np.random.rand(self.m, self.n)
        x = np.random.rand(self.n)
        b = np.random.rand(self.m)

        ell2 = 0.5*np.sum((A.dot(x) - b)**2)
        diff = 0.5*np.sum(lf.linear((A.dot(x) - b)**2)) - ell2
        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        tmp = A.dot(x) - b
        grad_ell2 = A.transpose().dot(tmp)
        diff = A.transpose().dot(lf.gradient_linear(tmp**2) * tmp) - grad_ell2
        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

    def test_soft_l1(self):

        b = np.random.rand(self.m)

        soft_l1 = np.zeros_like(b)
        soft_l1_grad = np.zeros_like(b)

        for i in xrange(0, self.m):
            e = b[i]
            soft_l1[i] = 2*(np.sqrt(1+e)-1)
            soft_l1_grad[i] = 1./np.sqrt(1+e)

        diff = lf.soft_l1(b) - soft_l1

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff_grad = lf.gradient_soft_l1(b) - soft_l1_grad
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad), decimals=self.accuracy), 0)

    def test_huber(self):

        b = np.random.rand(self.m)*10

        gamma = 1.3
        gamma2 = gamma * gamma

        huber = np.zeros_like(b)
        grad_huber = np.zeros_like(b)

        for i in range(0, self.m):
            e = b[i]
            if e < gamma2:
                huber[i] = e
                grad_huber[i] = 1
            else:
                huber[i] = 2*gamma*np.sqrt(e) - gamma2
                grad_huber[i] = gamma/np.sqrt(e)

        diff = lf.huber(b, gamma=gamma) - huber

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff_grad = lf.gradient_huber(b, gamma=gamma) - grad_huber
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad), decimals=self.accuracy), 0)

    def test_cauchy(self):

        b = np.random.rand(self.m)

        cauchy = np.zeros_like(b)
        cauchy_grad = np.zeros_like(b)

        for i in xrange(0, self.m):
            e = b[i]
            cauchy[i] = np.log(1+e)
            cauchy_grad[i] = 1./(1+e)

        diff = lf.cauchy(b) - cauchy

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff_grad = lf.gradient_cauchy(b) - cauchy_grad
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad), decimals=self.accuracy), 0)

    def test_arctan(self):

        b = np.random.rand(self.m)

        arctan = np.zeros_like(b)
        arctan_grad = np.zeros_like(b)

        for i in xrange(0, self.m):
            e = b[i]
            arctan[i] = np.arctan(e)
            arctan_grad[i] = 1. / (1 + e*e)

        diff = lf.arctan(b) - arctan

        self.assertEqual(np.around(
            np.linalg.norm(diff), decimals=self.accuracy), 0)

        diff_grad = lf.gradient_arctan(b) - arctan_grad
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad), decimals=self.accuracy), 0)

    def test_show_curves(self):

        M = 20
        steps = 100*M
        residual = np.linspace(-M, M, steps)

        residual2 = residual**2

        losses = ["linear", "soft_l1", "huber"]
        loss = []
        grad_loss = []
        jac = []
        labels = []

        loss.append(lf.linear(residual2))
        grad_loss.append(lf.gradient_linear(residual2))
        jac.append(lf.gradient_linear(residual2)*residual)
        labels.append("linear")

        loss.append(lf.soft_l1(residual2))
        grad_loss.append(lf.gradient_soft_l1(residual2))
        jac.append(lf.gradient_soft_l1(residual2)*residual)
        labels.append("soft_l1")

        for gamma in (1, 1.345, 5, 10, 15):
            loss.append(lf.huber(residual2, gamma=gamma))
            grad_loss.append(lf.gradient_huber(residual2, gamma=gamma))
            jac.append(lf.gradient_huber(residual2, gamma=gamma)*residual)
            labels.append("huber(" + str(gamma) + ")")

        ph.show_curves(loss, x=residual, labels=labels,
                       title="losses rho(x^2)")
        ph.show_curves(grad_loss, x=residual, labels=labels,
                       title="gradient losses rho'(x^2)")
        ph.show_curves(jac, x=residual, labels=labels,
                       title="jacobian rho'(x^2)*x")

    def test_cost_from_residual_linear(self):

        loss = "linear"

        def f(x):
            nda = np.zeros(3)
            nda[0] = x[0]**2 - 3*x[1]**3 + 5
            nda[1] = 2*x[0] + x[1]**2 - 1
            nda[2] = x[0] + x[1]
            return nda

        def df(x):
            nda = np.zeros((3, 2))
            nda[0, 0] = 2*x[0]
            nda[1, 0] = 2
            nda[2, 0] = 1

            nda[0, 1] = -9*x[1]**2
            nda[1, 1] = 2*x[1]
            nda[2, 1] = 1
            return nda

        X0, X1 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(0, 10, 0.2))
        points = np.array([X1.flatten(), X0.flatten()])

        cost_gd = lambda x: 0.5 * np.sum(f(x)**2)
        grad_cost_gd = lambda x: np.sum(f(x)[:, np.newaxis]*df(x), 0)

        for i in range(points.shape[1]):
            point = points[:, i]
            diff_cost = cost_gd(point) - \
                lf.get_cost_from_residual(f(point), loss=loss)
            diff_grad = grad_cost_gd(point) - \
                lf.get_gradient_cost_from_residual(
                    f(point), df(point), loss=loss)

            self.assertEqual(np.around(
                np.linalg.norm(diff_cost), decimals=self.accuracy), 0)
            self.assertEqual(np.around(
                np.linalg.norm(diff_grad), decimals=self.accuracy), 0)
