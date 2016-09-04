##-----------------------------------------------------------------------------
# \file Optimization_nnls.py
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Aug 2016
#

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear
from scipy.optimize import minimize
from scipy.optimize import least_squares
import scipy.optimize
from scipy.optimize import nnls

from nilearn.decoding.fista import mfista
from nilearn.decoding.proximal_operators import (
    _prox_tvl1,
    _prox_l1)
from nilearn.decoding.objective_functions import (
    _squared_loss,
    _logistic,
    _gradient_id,
    _squared_loss_grad,
    _logistic_loss_lipschitz_constant,
    _tv_l1_from_gradient,
    spectral_norm_squared)

"""
Functions
"""
## Get forward operator as matrix, i.e. blurring operator of image
def get_matrix_A(n, sigma):
    x = np.linspace(0,1,n)
    X,Y = np.meshgrid(x,x)

    return 1/(n*np.sqrt(2*np.pi)*sigma)*np.exp(-(Y-X)**2/(2*sigma**2))

def get_matrix_D(n):
    kernel = np.array([-1,1])
    D = np.zeros((n,n))

    for i in range(0, n-1):
        D[i,i:i+2] = kernel
    D[-1,-1] = -1

    return D
        



def lsmr_with_init(A,b,x0):
    r0 = b - scipy.sparse.linalg.aslinearoperator(A).matvec(x0)
    deltax_pack = scipy.sparse.linalg.lsmr(A,r0)
    return x0 + deltax_pack[0]

def admm_nnlsqr(A,b):
    # ADMM parameter
    # Optimal choice product of the maximum and minimum singular values
    # Heuristic choice: mean of singular values, i.e. squared Frobenius norm over dimension
    rho = np.dot(A.ravel(),A.ravel()) / A.shape[1]
    sqrt_half_rho = np.sqrt(rho/2)
    print 'sqrt_half_rho=',sqrt_half_rho
    #sqrt_half_rho = 100

    
    # initialisation
    zero_init = False
    if zero_init:
        x_pack = (np.zeros(A.shape[1]),0)
        z = np.zeros(A.shape[1])
        u = np.zeros(A.shape[1])
    else:
        # from x=z=u=0 the first iteration is normally a projection of
        # the unconstrained damped least squares. Let's forget the damping and check
        # whether we need to constrain things
        x_pack = scipy.sparse.linalg.lsmr(A,b)

        if np.all(x_pack[0]>0):
            # no constraint needed
            return x_pack[0]
    
        z = np.clip(x_pack[0],0,np.inf)
        u = x_pack[0]-z

    # construct the damped least squares matrix
    # todo try and use directly the damped version of lsmr
    Adamped = scipy.sparse.linalg.LinearOperator(
        A.shape+np.array([A.shape[1], 0]),
        matvec = lambda y: np.concatenate([ A.dot(y), sqrt_half_rho*y ]),
        rmatvec = lambda y: y[0:A.shape[0]].dot(A) + sqrt_half_rho*y[A.shape[0]:],
        )

    tol = 1e-5
    max_iter = 5000
    diff = np.inf
    iter = 0
    while ( diff>tol and iter<max_iter ):
        iter += 1
        xold = x_pack[0]
        #x_pack = scipy.sparse.linalg.lsmr( Adamped, np.concatenate([ b, sqrt_half_rho*(z-u) ]) )
        x_pack = (lsmr_with_init( Adamped, np.concatenate([ b, sqrt_half_rho*(z-u) ]), xold ), 0)
        zold = z
        z = np.clip(x_pack[0]+u,0,np.inf)
        #diff = np.linalg.norm(z-zold)
        #diff = np.linalg.norm(x_pack[0]-xold)
        diff = np.linalg.norm(x_pack[0]-z)
        u += x_pack[0]-z
        print 'iter', iter, ' -- diff', diff
    
    return z


def tikhonov(A, b, alpha, reg_type="TK1", show=False):
        
    n = A.shape[0]

    A = aslinearoperator(A)
    alpha_sqrt = np.sqrt(alpha)

    if reg_type in ["TK0"]:
        G = aslinearoperator(np.eye(n))
    else:
        G = aslinearoperator(get_matrix_D(n))

    A_fw = lambda x: np.concatenate((A.matvec(x), alpha_sqrt*G.matvec(x)))
    A_bw = lambda y: A.rmatvec(y[0:n]) + alpha_sqrt*G.rmatvec(y[n:])
    A_augmented = LinearOperator((2*n,n), matvec=A_fw, rmatvec=A_bw)

    b_augmented = np.zeros(2*n)
    b_augmented[0:n] = b

    return lsmr(A_augmented, b_augmented, show=show)[0]


def tvl2(A, b, alpha=0.05, rho=1, ADMM_iter=10, show=False):

    n = A.shape[0]

    A = aslinearoperator(A)
    D = aslinearoperator(get_matrix_D(n))

    rho_sqrt = np.sqrt(rho)
    alpha_over_rho = alpha/rho

    v = D.dot(b)
    w = np.zeros_like(b)

    A_fw = lambda x: np.concatenate((A.matvec(x), rho_sqrt*D.matvec(x)))
    A_bw = lambda y: A.rmatvec(y[0:n]) + rho_sqrt*D.rmatvec(y[n:])
    A_augmented = LinearOperator((2*n,n), matvec=A_fw, rmatvec=A_bw)
    b_augmented = np.concatenate((b, rho_sqrt*(v-w)))


    if show:
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(1,1,1)

    for i_ADMM in range(0, ADMM_iter):
        x = lsmr(A_augmented, b_augmented, show=show)[0]

        Dx = D.dot(x)
        t = Dx + w
        v = np.maximum(np.abs(t) - alpha_over_rho, 0)*np.sign(t)

        w = w + Dx - v

        b_augmented[n:] = rho_sqrt*(v-w)

        if show:
            plt.plot(x,label="x (iter="+str(i_ADMM)+")")
            legend = ax.legend(loc="upper right")
    return x


def LBFGSB(A, b, alpha=0.05, show=False):
    n = A.shape[0]
    alpha_sqrt = np.sqrt(alpha)

    A = aslinearoperator(A)
    D = aslinearoperator(get_matrix_D(n))

    A_fw = lambda x: np.concatenate((A.matvec(x), alpha_sqrt*D.matvec(x)))
    A_bw = lambda y: A.rmatvec(y[0:n]) + alpha_sqrt*D.rmatvec(y[n:])

    b_augmented = np.zeros(2*n)
    b_augmented[0:n] = b

    ## Set cost function and its jacobian
    fun = lambda x: 0.5*np.sum((A_fw(x) - b_augmented)**2)
    jac = lambda x: A_bw(A_fw(x)-b_augmented)


    return minimize(method='L-BFGS-B', fun=fun, jac=jac, x0=b, options={'disp': show}).x


def test_mfista(A, b):
    alpha = 0.1
    alpha_ = alpha * A.shape[0]
    l1_ratio = .2
    l1_weight = alpha_ * l1_ratio
    l1_weight = 0.1

    print("l1_weight = " + str(l1_weight))
    lipschitz_constant = np.linalg.norm(A.A,2)**2 # 2-norm: largest sing. value
    # lipschitz_constant = np.linalg.norm(A.A,'fro')**2 # 2-norm: largest sing. value
    # lipschitz_constant = np.linalg.norm(A.A,'')**2
    # lipschitz_constant = 1e10
    # lipschitz_constant = 1e4

    f1 = lambda x: _squared_loss(A, b, x, compute_energy=True, compute_grad=False)
    f1_grad = lambda x: _squared_loss(A, b, x, compute_energy=False, compute_grad=True)

    ## Seems to work for "good choice" of lipschitz_constant
    f2_prox = lambda x, l, *args, **kwargs: (_prox_l1(x, l * l1_weight), dict(converged=True))
    total_energy = lambda x: f1(x) + l1_weight * np.sum(np.abs(x))

    ## Does not seem to work
    # f2_prox = lambda x, l, *args, **kwargs: _prox_tvl1(x, l * l1_weight)
    # total_energy = lambda x: f1(x) + l1_weight * _tv_l1_from_gradient(_gradient_id(x))
    
    best_x, objective, init = mfista(
        f1_grad, f2_prox, total_energy, lipschitz_constant, A.shape[1], tol=1e-12, max_iter=500)

    return best_x



"""
Main Function
"""
if __name__ == '__main__':

    noise_level = 0.05                 ## Noise level for test image, default = 0.01

    sigma = 0.02
    n = 100

    disp_iterations = True

    ## Define solution
    x = np.zeros(n)
    x[12] = 0.5
    x[20:28] = 1.5
    x[32:36] = 2
    x[50:64] = 1

    ## Sinusoid as solution
    # x = np.sin(np.arange(0,n)/2)

    A = get_matrix_A(n, sigma)
    B = get_matrix_A(n, sigma)

    A = aslinearoperator(A)

    y_blur = A.dot(x)
    y_blur_noise = y_blur + noise_level*np.random.randn(n)

    # b = y_blur
    b = y_blur_noise

    
    x_lsmr = lsmr(A, b, show=disp_iterations)[0]
    x_lsqr = lsqr(A, b, show=disp_iterations)[0]
    x_lsqr = lsqr(A, b, show=disp_iterations)[0]
    x_LBFGSB = LBFGSB(A, b, alpha=0.05, show=disp_iterations)
    x_TK0 = tikhonov(A, b, alpha=0.1, reg_type="TK0", show=disp_iterations)
    x_TK1 = tikhonov(A, b, alpha=0.1, reg_type="TK1", show=disp_iterations)
    x_TVL2 = tvl2(A, b, alpha=0.05, rho=0.1, ADMM_iter=20)

    # x_admm_nnlsqr = admm_nnlsqr(A, b)
    x_mFISTA = test_mfista(A, b)



    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1,1,1)

    plt.plot(x,label="x", linewidth=3)
    plt.plot(b, label="b (observed)", linestyle="--", linewidth=3)

    # plt.plot(x_lsmr, label="x (lsmr)")
    # plt.plot(x_lsqr, label="x (lsqr)")
    plt.plot(x_LBFGSB, label="x (L-BFGS-B)")
    # plt.plot(x_admm_nnlsqr, label="x (admm_nnlsqr)")
    plt.plot(x_TK0, label="x (TK0)")
    plt.plot(x_TK1, label="x (TK1)")
    plt.plot(x_TVL2, label="x (TVL2)")
    plt.plot(x_mFISTA, label="x (mFISTA)")


    legend = ax.legend(loc="upper right")
    # plt.draw()
    plt.show(block=False)

    D = get_matrix_D(n)