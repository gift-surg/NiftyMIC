#!/usr/bin/env python
import numpy as np
import scipy.optimize

from nilearn.decoding.fista import mfista
from nilearn.decoding.proximal_operators import _prox_l1
from nilearn.decoding.objective_functions import (
    _squared_loss,
    _logistic,
    _squared_loss_grad,
    _logistic_loss_lipschitz_constant,
    spectral_norm_squared)

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

def test_mfista(A, b):
    alpha = .01
    alpha_ = alpha * A.shape[0]
    l1_ratio = .2
    l1_weight = alpha_ * l1_ratio

    f1 = lambda x: _squared_loss(A, b, x, compute_energy=True, compute_grad=False)
    f1_grad = lambda x: _squared_loss(A, b, x, compute_energy=False, compute_grad=True)
    f2_prox = lambda x, l, *args, **kwargs: (_prox_l1(x, l * l1_weight),
                                             dict(converged=True))

    total_energy = lambda x: f1(x) + l1_weight * np.sum(np.abs(x))
    best_x, objective, init = mfista(
        f1_grad, f2_prox, total_energy, 1, A.shape[1], tol=1e-12,
        max_iter=100)

    return best_x


A = np.array([[60, 90, 120],[30, 120, 90],[0, 12, 8],[-45, 8, 7]])
b = np.array([67.5, -60, 8, 0.4])

x_nnls, rnorm_nnls = scipy.optimize.nnls(A,b)
print 'x   (nnls) =', x_nnls
print 'norm       = ', np.linalg.norm(A.dot(x_nnls)-b)

x_lstsq_pack = np.linalg.lstsq(A,b)
x_lstsq = x_lstsq_pack[0]
print 'x  (lstsq) =', x_lstsq
print 'norm       = ', np.linalg.norm(A.dot(x_lstsq)-b)

x_lsqlin_pack = scipy.optimize.lsq_linear(A,b,bounds=(0,np.inf),lsq_solver='lsmr')
x_lsqlin = x_lsqlin_pack.x
print 'x (lsqlin) =', x_lsqlin
print 'norm       = ', np.linalg.norm(A.dot(x_lsqlin)-b), 'niter=', x_lsqlin_pack.nit

x_lsmr_pack = scipy.sparse.linalg.lsmr(A,b)
x_lsmr = x_lsmr_pack[0]
print 'x   (lsmr) =', x_lsmr
print 'norm       = ', np.linalg.norm(A.dot(x_lsmr)-b)

x_lsqr_pack = scipy.sparse.linalg.lsqr(A,b)
x_lsqr = x_lsqr_pack[0]
print 'x   (lsqr) =', x_lsqr
print 'norm       = ', np.linalg.norm(A.dot(x_lsqr)-b)

# x_nnlsqr = admm_nnlsqr(A,b)
# print 'x (nnlsqr) =', x_nnlsqr
# print 'norm       = ', np.linalg.norm(A.dot(x_nnlsqr)-b)

x_mfista = test_mfista(A,b)
print 'x (mfista) =', x_mfista
print 'norm       = ', np.linalg.norm(A.dot(x_mfista)-b)