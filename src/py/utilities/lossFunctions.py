##
# \file lossFunctions.py
# \brief      Definition of cost, residual and loss functions according to \p
#             https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.least_squares.html
#
# The overall cost function is defined as
# \f[ C:\,\mathbb{R}^n \rightarrow \mathbb{R}_{\ge 0},\,\vec{x} \mapsto
# C(\vec{x}) := \frac{1}{2} \sum_{i=0}^{m-1} \rho( f_i(\vec{x})^2 )
# \f] with the loss function
# \f$\rho:\,\mathbb{R}\rightarrow\mathbb{R}_{\ge0}\f$ and the residual
# \f$ \vec{f}:\, \mathbb{R}^n \rightarrow
# \mathbb{R}^m,\,\vec{x}\mapsto\vec{f}(\vec{x})
# = \big(f_0(\vec{x}),\,f_1(\vec{x}),\dots, f_{m-1}(\vec{x})\big)
# \f$
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       May 2017
#

import numpy as np


##
# Gets the cost \f$C(\vec{x})\f$ from the residual \f$\vec{f}(\vec{x})\f$.
# \date       2017-07-14 11:42:18+0100
#
# \param      f      residual f, m-dimensional numpy array
# \param      loss   Choice for loss function rho
#
# \return     The cost from residual as scalar value >= 0
#
def get_ell2_cost_from_residual(f, loss="linear"):
    cost = 0.5 * np.sum(get_loss[loss](f**2))
    return cost


##
# Gets the gradient
# \f$
# \nabla C\f$ of the cost function given the residual
# \f$\vec{f}(\vec{x})\f$ and its Jacobian
# \f$\frac{d\vec{f}}{d\vec{x}}(\vec{x})\f$.
# \date       2017-07-14 11:30:11+0100
#
# \param      f      residual f, m-dimensional numpy array
# \param      jac_f  Jacobian of residual f, (m x n)-dimensional numpy array
# \param      loss   Choice for loss function rho
#
# \return     The gradient of the cost from residual as n-dimensional numpy
#             array
#
def get_gradient_ell2_cost_from_residual(f, jac_f, loss="linear"):
    grad = np.sum((get_gradient_loss[loss](f**2) * f)[:, np.newaxis] * jac_f,
                  axis=0)
    return grad


def linear(f):
    return f


def gradient_linear(f):
    return np.ones_like(f).astype(np.float64)


def soft_l1(f):
    return 2. * (np.sqrt(1.+f) - 1.)


def gradient_soft_l1(f):
    return 1. / np.sqrt(1.+f)


def huber(f, gamma=1.345):
    gamma = float(gamma)
    gamma2 = gamma * gamma
    return np.where(f < gamma2, f, 2.*gamma*np.sqrt(f)-gamma2)


def gradient_huber(f, gamma=1.345):
    gamma = float(gamma)
    gamma2 = gamma * gamma
    return np.where(f < gamma2, 1., gamma/np.sqrt(f))


def cauchy(f):
    return np.log1p(f)


def gradient_cauchy(f):
    return 1. / (1. + f)


def arctan(f):
    return np.arctan(f)


def gradient_arctan(f):
    return 1. / (1. + f**2)


get_loss = {
    "linear": linear,
    "soft_l1": soft_l1,
    "huber": huber,
    "cauchy": cauchy,
    "arctan": arctan,
}
get_gradient_loss = {
    "linear": gradient_linear,
    "soft_l1": gradient_soft_l1,
    "huber": gradient_huber,
    "cauchy": gradient_cauchy,
    "arctan": gradient_arctan,
}
