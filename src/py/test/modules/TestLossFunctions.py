## \file TestLossFunctions.py
#  \brief  Class containing unit tests for module Stack
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


## Import libraries 
import numpy as np
import unittest
import sys
import matplotlib.pyplot as plt

## Import modules
import utilities.lossFunctions as lf
# import utilities.SimpleITKHelper as sitkh
import utilities.PythonHelper as ph

from definitions import dir_test

## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestNiftyReg(unittest.TestCase):

    ## Specify input data
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
            np.linalg.norm(diff)
            , decimals = self.accuracy), 0 )

        diff = lf.gradient_linear(b) - 1
        self.assertEqual(np.around(
            np.linalg.norm(diff)
            , decimals = self.accuracy), 0 )

    def test_least_squares_linear(self):

        A = np.random.rand(self.m,self.n)
        x = np.random.rand(self.n)
        b = np.random.rand(self.m)

        ell2 = 0.5*np.sum((A.dot(x) - b)**2)
        diff = 0.5*np.sum(lf.linear((A.dot(x) - b)**2)) - ell2
        self.assertEqual(np.around(
            np.linalg.norm(diff)
            , decimals = self.accuracy), 0 )

        tmp = A.dot(x) - b
        grad_ell2 = A.transpose().dot(tmp)
        diff = A.transpose().dot( lf.gradient_linear(tmp**2) * tmp) - grad_ell2
        self.assertEqual(np.around(
            np.linalg.norm(diff)
            , decimals = self.accuracy), 0 )


    def test_soft_l1(self):

        b = np.random.rand(self.m)

        diff = np.zeros_like(b)
        diff_grad = np.zeros_like(b)

        for i in xrange(0, self.m):
            e = b[i]
            diff[i] = 2*(np.sqrt(1+e)-1)
            diff_grad[i] = 1./np.sqrt(1+e)

        diff = lf.soft_l1(b) - diff

        self.assertEqual(np.around(
            np.linalg.norm(diff)
            , decimals = self.accuracy), 0 )

        diff_grad = lf.gradient_soft_l1(b) - diff_grad
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad)
            , decimals = self.accuracy), 0 )


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
            np.linalg.norm(diff)
            , decimals = self.accuracy), 0 )

        diff_grad = lf.gradient_huber(b, gamma=gamma) - grad_huber
        self.assertEqual(np.around(
            np.linalg.norm(diff_grad)
            , decimals = self.accuracy), 0 )


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

        for gamma in (1,1.345,5,10,15):
            loss.append(lf.huber(residual2, gamma=gamma))
            grad_loss.append(lf.gradient_huber(residual2, gamma=gamma))
            jac.append(lf.gradient_huber(residual2, gamma=gamma)*residual)
            labels.append("huber(" + str(gamma) + ")")

        ph.show_curves(loss, x=residual, labels=labels, title="losses rho(x^2)")
        ph.show_curves(grad_loss, x=residual, labels=labels, title="gradient losses rho'(x^2)")
        ph.show_curves(jac, x=residual, labels=labels, title="jacobian rho'(x^2)*x")

