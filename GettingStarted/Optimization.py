## \file Optimization.py
#  \brief  SciPy optimization library applied to problems I can generalize to 
#       to my application. Several regularziation approaches are implemented,
#       like TK0 and TK1 in the standard and SPD version.
#       The cases used here don't consider the physical space to a large extent.
#       For scenarios which involve different oritentations, see
#       ITK_ReconstructVolume.py
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date March 2016

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import leastsq
from scipy import ndimage

import SimpleITK as sitk
import numpy as np
import unittest
import matplotlib.pyplot as plt


import sys
import time
sys.path.append("../src")

import SimpleITKHelper as sitkh

"""
Functions
"""
## Compute Gaussian kernel for 2D and 3D scenario.
#  Mean of multivariate Gaussian is assumed to be zero
#  \param[in] Cov Variance-covariance matrix
#  \param[in] alpha Cut-off distance
#  \param[in] scaling Scaling incorporated in covariance matrix
def get_gaussian_kernel(Cov, alpha, scaling=None):
    Sigma = np.sqrt(Cov.diagonal())
    dim = Cov.shape[0]
    alpha = np.array(alpha)

    origin = np.zeros((dim,1))

    if scaling is None:
        scaling = np.ones(dim)

    S = np.diag(scaling)


    if dim == 2:
        ## Generate intervals for x and y based on cut-off distance given by
        ## Sigma, alpha and scaling
        [x_max, y_max] = np.ceil(Sigma*alpha/scaling)

        step = 1
        x_interval = np.arange(-x_max, x_max+step,step)
        y_interval = np.arange(-y_max, y_max+step,step)

        ## Generate arrays of 2D points bearing in mind that nifti-nda.shape = (y,x)-coord
        [X,Y] = np.meshgrid(x_interval, y_interval, indexing='ij')    # 'ij' yields vertical x-coordinate for image!
        points = np.array([Y.flatten(), X.flatten()])

        ## Compute scaled, inverse covariance matrix
        Cov_scale_inv = S.dot(np.linalg.inv(Cov)).dot(S)

        ## Compute Gaussian weights
        values = np.sum( (points-origin)*Cov_scale_inv.dot(points-origin), 0)
        kernel = np.exp( -0.5*values );
        kernel = kernel/np.sum(kernel);

        ## Reshape kernel
        kernel = kernel.reshape(x_interval.size, y_interval.size)

    if dim == 3:
        ## Generate intervals for x and y based on cut-off distance given by
        ## Sigma, alpha and scaling
        [x_max, y_max, z_max] = np.ceil(Sigma*alpha/scaling)

        step = 1
        x_interval = np.arange(-x_max, x_max+step,step)
        y_interval = np.arange(-y_max, y_max+step,step)
        z_interval = np.arange(-z_max, z_max+step,step)

        ## Generate arrays of 3D points bearing in mind that nifti-nda.shape = (z,y,x)-coord
        [X,Y,Z] = np.meshgrid(x_interval, y_interval, z_interval, indexing='ij')    # 'ij' yields vertical x-coordinate for image!
        points = np.array([Z.flatten(), Y.flatten(), X.flatten()])

        ## Compute scaled, inverse covariance matrix
        Cov_scale_inv = S.dot(np.linalg.inv(Cov)).dot(S)

        ## Compute Gaussian weights
        values = np.sum( (points-origin)*Cov_scale_inv.dot(points-origin), 0)
        kernel = np.exp( -0.5*values );
        kernel = kernel/np.sum(kernel);

        ## Reshape kernel
        kernel = kernel.reshape(x_interval.size, y_interval.size, z_interval.size)

    return kernel


## Compute forward difference quotient in x-direction to differentiate
#  array with array = array[(z,)y,x], i.e. the 'correct' direction
#  by viewing the resulting nifti-image differentiation. The resulting kernel
#  can be used via getDx(kernel, nda) to differentiate image
#  \param[in] dim spatial dimension
#  \return kernel kernel for dim dimensional differentiation in x
def get_forward_difference_x_kernel(dim):
    if dim is 2:
        ## kernel = np.zeros((y,x))
        kernel = np.zeros((1,2))
        kernel[:] = np.array([1,-1])

    if dim is 3:
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,1,2))
        kernel[:] = np.array([1,-1])

    return kernel


## Compute backward difference quotient in x-direction to differentiate
#  array with array = array[(z,)y,x], i.e. the 'correct' direction
#  by viewing the resulting nifti-image differentiation. The resulting kernel
#  can be used via getDx(kernel, nda) to differentiate image
#  \param[in] dim spatial dimension
#  \return kernel kernel for dim dimensional differentiation in x
def get_backward_difference_x_kernel(dim):
    if dim is 2:
        ## kernel = np.zeros((y,x))
        kernel = np.zeros((1,3))
        kernel[:] = np.array([0,1,-1])

    if dim is 3:
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,1,3))
        kernel[:] = np.array([0,1,-1])

    return kernel


## Compute forward difference quotient in y-direction to differentiate
#  array with array = array[(z,)y,x], i.e. the 'correct' direction
#  by viewing the resulting nifti-image differentiation. The resulting kernel
#  can be used via getDx(kernel, nda) to differentiate image
#  \param[in] dim spatial dimension
#  \return kernel kernel for dim dimensional differentiation in y
def get_forward_difference_y_kernel(dim):
    if dim is 2:
        ## kernel = np.zeros((y,x))
        kernel = np.zeros((2,1))
        kernel[:] = np.array([[1],[-1]])

    if dim is 3:
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,2,1))
        kernel[:] = np.array([[1],[-1]])

    return kernel


## Compute backward difference quotient in y-direction to differentiate
#  array with array = array[(z,)y,x], i.e. the 'correct' direction
#  by viewing the resulting nifti-image differentiation. The resulting kernel
#  can be used via getDx(kernel, nda) to differentiate image
#  \param[in] dim spatial dimension
#  \return kernel kernel for dim dimensional differentiation in y
def get_backward_difference_y_kernel(dim):
    if dim is 2:
        ## kernel = np.zeros((y,x))
        kernel = np.zeros((3,1))
        kernel[:] = np.array([[0],[1],[-1]])

    if dim is 3:
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,3,1))
        kernel[:] = np.array([[0],[1],[-1]])

    return kernel


## Compute forward difference quotient in z-direction to differentiate
#  array with array = array[z,y,x], i.e. the 'correct' direction
#  by viewing the resulting nifti-image differentiation. The resulting kernel
#  can be used via getDx(kernel, nda) to differentiate image
#  \param[in] dim spatial dimension
#  \return kernel kernel for dim dimensional differentiation in z
def get_forward_difference_z_kernel(dim):
    if dim is 2:
        raise ValueError("Error: Differentiation in z-direction requires a 3-dimensional space!")

    if dim is 3:
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((2,1,1))
        kernel[:] = np.array([[[1]],[[-1]]])

    return kernel


## Compute backward difference quotient in y-direction to differentiate
#  array with array = array[(z,)y,x], i.e. the 'correct' direction
#  by viewing the resulting nifti-image differentiation. The resulting kernel
#  can be used via getDx(kernel, nda) to differentiate image
#  \param[in] dim spatial dimension
#  \return kernel kernel for dim dimensional differentiation in y
def get_backward_difference_z_kernel(dim):
    if dim is 2:
        raise ValueError("Error: Differentiation in z-direction requires a 3-dimensional space!")

    if dim is 3:
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((3,1,1))
        kernel[:] = np.array([[[0]],[[1]],[[-1]]])

    return kernel


## Compute Laplacian kernel to differentiate
#  array with array = array[(z,)y,x], i.e. the 'correct' direction
#  by viewing the resulting nifti-image differentiation. The resulting kernel
#  can be used via getDx(kernel, nda) to differentiate image
#  \param[in] dim spatial dimension
#  \return kernel kernel for Laplacian operation
def get_laplacian_kernel(dim):
    if dim is 2:
        ## kernel = np.zeros((y,x))
        kernel = np.zeros((3,3))
        kernel[:] = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])

    if dim is 3:
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((3,3,3))
        kernel[0,1,1] = 1
        kernel[1,:,:] = np.array([[0, 1, 0],[1, -6, 1],[0, 1, 0]])
        kernel[2,1,1] = 1
        
    return kernel


## Compute blurred image with added noise
#  \param[in] image_sitk sitk::Image with shall be distorted
#  \param[in] Cov Variance-covariance matrix used for Gaussian blurring
#  \param[in] alpha Cut-off distance for Gaussian blurring
#  \param[in] multiplicative_noise Indicates whether multiplicative or additve noise shall be added
def get_blurred_noisy_image(image_sitk, Cov, alpha, noise_level=0.01, multiplicative_noise=True):

    dim = image_sitk.GetDimension()
    spacing = np.array(image_sitk.GetSpacing())
    S = np.diag(spacing)

    ## Get Gaussian kernel
    kernel = get_gaussian_kernel(Cov, alpha, spacing)

    ## Blur image
    nda = sitk.GetArrayFromImage(image_sitk)
    observed_nda = getA(nda, kernel)

    ## Add noise
    if multiplicative_noise:
        if dim is 2:
            noise = noise_level * np.random.rand(nda.shape[0], nda.shape[1]) * nda
        else:
            noise = noise_level * np.random.rand(nda.shape[0], nda.shape[1], nda.shape[2]) * nda

    else:
        Sigma = np.sqrt(Cov.diagonal())
        if dim is 2:
            noise = noise_level * np.max(np.abs(nda)) * np.random.rand(nda.shape[0], nda.shape[1])
        else:
            noise = noise_level * np.max(np.abs(nda)) * np.random.rand(nda.shape[0], nda.shape[1], nda.shape[2])

    observed_nda += noise

    ## Create sitk image
    image_observed_sitk = sitk.GetImageFromArray(observed_nda)
    image_observed_sitk.CopyInformation(image_sitk)

    return image_observed_sitk


## Compute forward operation A(x), i.e. blurring of image.
#  The boundary condition is chosen based on check_adjoint_Gaussian_blurring_boundary_conditions.
#  \param[in] kernel Gaussian kernel for blurring
#  \param[in] nda image array which will get blurred
#  \return Gaussian blurred data array
def getA(nda, kernel):
    return ndimage.convolve(nda, kernel, mode='wrap')


## Compute adjoint operation A^*(y).
#  The boundary condition is chosen based on check_adjoint_Gaussian_blurring_boundary_conditions.
#  \param[in] kernel Gaussian kernel for blurring
#  \param[in] nda image array which will get blurred
#  \return Adjoint Gaussian blurred data array
def getAT(nda, kernel):
    return ndimage.convolve(nda, kernel, mode='wrap')


## Compute derivative of array based on given kernel
#  The boundary condition is chosen based on check_adjoint_differentiation_boundary_conditions.
#  \param[in] kernel_D kernel defining the differentiation
#  \param[in] nda image array which will get differentiated
#  \return Derivative of image determined by kernel
def getDx(nda, kernel_D):
    return ndimage.convolve(nda, kernel_D, mode='wrap')


## Compute derivative of array based on given kernel
#  The boundary condition is chosen based on check_adjoint_differentiation_boundary_conditions.
#  \param[in] kernel_D_adj adjoint kernel defining the differentiation
#  \param[in] nda image array which will get differentiated
#  \return Adjoint derivative of image determined by kernel
def getDTx(nda, kernel_D_adj):
    return ndimage.convolve(nda, kernel_D_adj, mode='wrap')

## Compute A'A(x), A being the Gaussian blurring
#  \param[in] nda image array which will get blurred
#  \param[in] kernel Gaussian kernel for blurring
def getATA(nda, kernel):
    return getAT(getA(nda,kernel), kernel)


## Compute D'D(x), D being the differential operator
#  \param[in] nda image array which will get differentiated
#  \param[in] kernel_Dx kernel used for differentiation in x-coordinate
#  \param[in] kernel_Dx_adj adjoint kernel of kernel_Dx
#  \param[in] kernel_Dy kernel used for differentiation in y-coordinate
#  \param[in] kernel_Dy_adj adjoint kernel of kernel_Dy
#  \param[in] kernel_Dz kernel used for differentiation in z-coordinate (optional)
#  \param[in] kernel_Dz_adj adjoint kernel of kernel_Dz (optional)
def getDTDx(nda, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz=None, kernel_Dz_adj=None):
    if kernel_Dz is None:

        ## Compute [Dx]_x and [Dx]_y
        Dx_x = getDx(nda, kernel_Dx)
        Dy_x = getDx(nda, kernel_Dy)

        ## Compute [D'(Dx)]_x and [D'(Dx)]_y
        DTDx_x = getDx(Dx_x, kernel_Dx_adj)
        DTDy_x = getDx(Dy_x, kernel_Dy_adj)

        ## D'Dx = [D'(Dx)]_x + [D'(Dx)]_y (i.e. 'Laplace')
        DTDx = DTDx_x + DTDy_x

    else:
        ## Compute [Dx]_x, [Dx]_y and [Dx]_z
        Dx_x = getDx(nda, kernel_Dx)
        Dy_x = getDx(nda, kernel_Dy)
        Dz_x = getDx(nda, kernel_Dz)

        ## Compute [D'(Dx)]_x, [D'(Dx)]_y and  [D'(Dx)]_z
        DTDx_x = getDx(Dx_x, kernel_Dx_adj)
        DTDy_x = getDx(Dy_x, kernel_Dy_adj)
        DTDz_x = getDx(Dz_x, kernel_Dz_adj)

        ## D'Dx = [D'(Dx)]_x + [D'(Dx)]_y  + [D'(Dx)]_z (i.e. 'Laplace')
        DTDx = DTDx_x + DTDy_x + DTDz_x

    return DTDx

## Compute cost function 
#           J0(x) = 0.5*|| g - Ax ||^2 + 0.5*alpha*||x||^2
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] g_vec     g as vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \return    J0(x), scalar value
def TK0(x_vec, g_vec, kernel, alpha, shape):

    ## Get dimension to reshape
    # dim2 = x_vec.size
    # dim = np.round(np.sqrt(dim2)).astype('uint')

    ## Reshape so that blurring filter can be applied
    x = x_vec.reshape(shape)
    
    ## Return J0(x) = 0.5*|| g - Ax ||^2 + 0.5*alpha*||x||^2
    return 0.5*np.sum( (g_vec-getA(x,kernel).flatten())**2 ) + 0.5*alpha*x_vec.dot(x_vec)


## Compute gradient of cost function J0(x), i.e.
#           grad J0(x) = A'(Ax - g) + alpha*x
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] ATg_vec   A'g as vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \return    grad J0(x), vector array
def TK0_grad(x_vec, ATg_vec, kernel, alpha, shape):

    ## Reshape so that blurring filter can be applied
    x = x_vec.reshape(shape)

    ## Return grad J0(x) = A'(Ax - g) + alpha*x
    return  getATA(x,kernel).flatten() - ATg_vec + alpha*x_vec


## Compute hessian matrix of cost function J0(x) applied on p, i.e.
#           (Hessian J0(x))p = A'Ap + alpha*p
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] p_vec     point on which hessian is applied, vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data arraya
#  \return    (Hessian J0(x))p, vector array
def TK0_hess_p(x_vec, p_vec, kernel, alpha, shape):

    ## Reshape so that blurring filter can be applied
    p = p_vec.reshape(shape)

    ## Return hess_p = A'A(p) + alpha*p
    return getATA(p,kernel).flatten() + alpha*p_vec


## Compute cost function with SPD matrix
#           J0(x) = || (A'A + alpha)x - A'g ||^2
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] g_vec     g as vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \return    J0(x), scalar value
def TK0_SPD(x_vec, ATg_vec, kernel, alpha, shape):

    ## Get dimension to reshape
    # dim2 = x_vec.size
    # dim = np.round(np.sqrt(dim2)).astype('uint')

    ## Reshape so that blurring filter can be applied
    x = x_vec.reshape(shape)
    
    ## Compute op(x) := (A'A + alpha)x
    op_x = getATA(x,kernel).flatten() + alpha*x_vec

    ## Return || (A'A + alpha)x - A'g ||^2
    return np.sum((op_x-ATg_vec)**2)


## Compute gradient of cost function J0(x) with SPD matrix, i.e.
#           grad J0(x) = (A'A + alpha) ( (A'A + alpha)x - g )
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] op_ATg    (A'A + alpha)A'g as vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \return    grad J0(x), vector array
def TK0_SPD_grad(x_vec, op_ATg, kernel, alpha, shape):

    ## Reshape so that blurring filter can be applied
    x = x_vec.reshape(shape)

    ## Compute op(x) := (A'A + alpha)x
    op_x = getATA(x, kernel) + alpha*x
    
    ## Compute (A'A + alpha)(A'A + alpha)x = (A'A + alpha)op_x
    op_op_x = getATA(op_x,kernel).flatten() + alpha*op_x.flatten()

    ## Return grad J0(x) = (A'A + alpha) ( (A'A + alpha)x - g )
    return  op_op_x - op_ATg


## Compute hessian matrix of cost function J0(x) applied on p, i.e.
#           (Hessian J0(x))p = (A'A + alpha)(A'A + alpha)p
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] p_vec     point on which hessian is applied, vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \return    (Hessian J0(x))p, vector array
def TK0_SPD_hess_p(x_vec, p_vec, kernel, alpha, shape):

    ## Reshape so that blurring filter can be applied
    p = p_vec.reshape(shape)

    ## Compute op(x) := (A'A + alpha)x
    op_p = getATA(p, kernel) + alpha*p
    
    ## Return hess_p (A'A + alpha)(A'A + alpha)p = (A'A + alpha)op_p
    return getATA(op_p,kernel).flatten() + alpha*op_p.flatten()


## Compute cost function 
#           J1(x) = 0.5*|| g - Ax ||^2 + 0.5*alpha*||Dx||^2
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] g_vec     g as vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \param[in] kernel_Dx kernel used for differentiation in x-coordinate
#  \param[in] kernel_Dy kernel used for differentiation in y-coordinate
#  \param[in] kernel_Dz kernel used for differentiation in z-coordinate (optional)
#  \return    J1(x), scalar value
def TK1(x_vec, g_vec, kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dz=None):

    ## Get dimension to reshape
    # dim2 = x_vec.size
    # dim = np.round(np.sqrt(dim2)).astype('uint')

    ## Reshape so that blurring filter can be applied
    x = x_vec.reshape(shape)

    ## Compute Dx
    if kernel_Dz is None:
        Dx_x = getDx(x, kernel_Dx)
        Dy_x = getDx(x, kernel_Dy)

        Dx2 = Dx_x**2 + Dy_x**2
    else:
        Dx_x = getDx(x, kernel_Dx)
        Dy_x = getDx(x, kernel_Dy)
        Dz_x = getDx(x, kernel_Dz)

        Dx2 = Dx_x**2 + Dy_x**2 + Dz_x**2
    
    ## Return J1(x) = 0.5*|| g - Ax ||^2 + 0.5*alpha*||Dx||^2
    return 0.5*np.sum( (g_vec-getA(x,kernel).flatten())**2 ) + 0.5*alpha*np.sum( Dx2 )


## Compute gradient of cost function J1(x), i.e.
#           grad J1(x) = A'(Ax - g) + alpha*D'Dx
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] ATg_vec   A'g as vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \param[in] kernel_Dx kernel used for differentiation in x-coordinate
#  \param[in] kernel_Dx_adj adjoint kernel of kernel_Dx
#  \param[in] kernel_Dy kernel used for differentiation in y-coordinate
#  \param[in] kernel_Dy_adj adjoint kernel of kernel_Dy
#  \param[in] kernel_Dz kernel used for differentiation in z-coordinate (optional)
#  \param[in] kernel_Dz_adj adjoint kernel of kernel_Dz
#  \return    grad J1(x), vector array
def TK1_grad(x_vec, ATg_vec, kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz=None, kernel_Dz_adj=None):

    ## Reshape so that blurring filter can be applied
    x = x_vec.reshape(shape)

    ## Compute D'Dx
    DTDx = getDTDx(x, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)

    ## Return grad J1(x) = A'(Ax - g) + alpha*D'Dx
    return  getATA(x,kernel).flatten() - ATg_vec + alpha*DTDx.flatten()


## Compute hessian matrix of cost function J1(x) applied on p, i.e.
#           (Hessian J1(x))p = A'Ap + alpha*D'Dp
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] p_vec     point on which hessian is applied, vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \param[in] kernel_Dx kernel used for differentiation in x-coordinate
#  \param[in] kernel_Dx_adj adjoint kernel of kernel_Dx
#  \param[in] kernel_Dy kernel used for differentiation in y-coordinate
#  \param[in] kernel_Dy_adj adjoint kernel of kernel_Dy
#  \param[in] kernel_Dz kernel used for differentiation in z-coordinate (optional)
#  \param[in] kernel_Dz_adj adjoint kernel of kernel_Dz (optional)
#  \return    (Hessian J1(x))p, vector array
def TK1_hess_p(x_vec, p_vec, kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz=None, kernel_Dz_adj=None):

    ## Reshape so that blurring filter can be applied
    p = p_vec.reshape(shape)

    ## Compute D'Dp
    DTDp = getDTDx(p, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)

    ## Return hess_p = A'A(p) + alpha*D'Dp
    return getATA(p,kernel).flatten() + alpha*DTDp.flatten()


## Compute cost function with SPD matrix
#           J1(x) = || A'g - (A'A + alpha*D'D)x ||^2
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] g_vec     g as vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \param[in] kernel_Dx kernel used for differentiation in x-coordinate
#  \param[in] kernel_Dx_adj adjoint kernel of kernel_Dx
#  \param[in] kernel_Dy kernel used for differentiation in y-coordinate
#  \param[in] kernel_Dy_adj adjoint kernel of kernel_Dy
#  \param[in] kernel_Dz kernel used for differentiation in z-coordinate (optional)
#  \param[in] kernel_Dz_adj adjoint kernel of kernel_Dz (optional)
#  \return    J1(x), scalar value
def TK1_SPD(x_vec, ATg_vec, kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz=None, kernel_Dz_adj=None):

    ## Reshape so that blurring filter can be applied
    x = x_vec.reshape(shape)

    ## Compute D'Dx
    DTDx = getDTDx(x, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)

    ## Return J1(x) = || A'g - (A'A + alpha*D'D)x ||^2
    return np.sum( ( ATg_vec - getATA(x,kernel).flatten() - alpha*DTDx.flatten() )**2 )


## Compute gradient of cost function with SPD matrix
#           grad J1(x) = (A'A + alpha*D'D)( (A'A + alpha*D'D)x - A'g )
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] op_ATg_vec (A'A + alpha*D'D)A'g, vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \param[in] kernel_Dx kernel used for differentiation in x-coordinate
#  \param[in] kernel_Dx_adj adjoint kernel of kernel_Dx
#  \param[in] kernel_Dy kernel used for differentiation in y-coordinate
#  \param[in] kernel_Dy_adj adjoint kernel of kernel_Dy
#  \param[in] kernel_Dz kernel used for differentiation in z-coordinate (optional)
#  \param[in] kernel_Dz_adj adjoint kernel of kernel_Dz (optional)
#  \return    grad J1(x), vector array
def TK1_SPD_grad(x_vec, op_ATg_vec, kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz=None, kernel_Dz_adj=None):

    ## Reshape so that blurring filter can be applied
    x = x_vec.reshape(shape)

    ## 1) Compute tmp1 = (A'A + alpha*D'D)x
    DTDx = getDTDx(x, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)
    ATAx = getATA(x, kernel)
    tmp1 = ATAx + alpha*DTDx

    ## 2) Compute tmp2 = (A'A + alpha*D'D)( (A'A + alpha*D'D)x ) = (A'A + alpha*D'D)tmp1
    DTDtmp1 = getDTDx(tmp1, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)
    ATAtmp1 = getATA(tmp1, kernel)
    tmp2 = ATAtmp1 + alpha*DTDtmp1
    
    ## Return grad J1(x) = (A'A + alpha*D'D)( (A'A + alpha*D'D)x - A'g )
    return tmp2.flatten() - op_ATg_vec


## Compute hessian matrix of cost function with SPD matrix
#           (Hessian J1(x))p = (A'A + alpha*D'D)(A'A + alpha*D'D)p
#  \param[in] x_vec     current estimate of solution, vector array
#  \param[in] p_vec     point on which hessian is applied, vector array
#  \param[in] kernel    kernel used for Gaussian blurring
#  \param[in] alpha     Regularization parameter
#  \param[in] shape     Dimension of image data array
#  \param[in] kernel_Dx kernel used for differentiation in x-coordinate
#  \param[in] kernel_Dx_adj adjoint kernel of kernel_Dx
#  \param[in] kernel_Dy kernel used for differentiation in y-coordinate
#  \param[in] kernel_Dy_adj adjoint kernel of kernel_Dy
#  \param[in] kernel_Dz kernel used for differentiation in z-coordinate (optional)
#  \param[in] kernel_Dz_adj adjoint kernel of kernel_Dz (optional)
#  \return    (Hessian J1(x))p, vector array
def TK1_SPD_hess_p(x_vec, p_vec, kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz=None, kernel_Dz_adj=None):

    ## Reshape so that blurring filter can be applied
    p = p_vec.reshape(shape)

    ## Compute tmp1 = (A'A + alpha*D'D)p
    DTDp = getDTDx(p, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)
    ATAp = getATA(p, kernel)
    tmp1 = ATAp + alpha*DTDp

    ## Return (Hessian J1(x))p = (A'A + alpha*D'D)( (A'A + alpha*D'D)p ) = (A'A + alpha*D'D)tmp1
    DTDtmp1 = getDTDx(tmp1, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)
    ATAtmp1 = getATA(tmp1, kernel)
    
    return (ATAtmp1 + alpha*DTDtmp1).flatten()

# def TK0_augmented(x_vec, g_vec, kernel, alpha, shape):

#     ## Reshape so that blurring filter can be applied
#     x = x_vec.reshape(shape)
    
#     ## Compute[A; sqrt(alpha)]*x - [g; 0]
#     val = np.zeros(2*x_vec.size)
#     val[0:x_vec.size] = getA(x,kernel).flatten() - g_vec
#     val[x_vec.size:] = np.sqrt(alpha)*x_vec

#     return val

## is supposed to return whole matrix! Not possible
# def TK0_augmented_grad(x_vec, g_vec, kernel, alpha, shape):

#     ## Reshape so that blurring filter can be applied
#     x = x_vec.reshape(shape)

    ## Return [A; ]


## Use scipy.optimize.minimize to get an approximate solution
#  \param[in] fun   objective function to minimize, returns scalar value
#  \param[in] jac   jacobian of objective function, returns vector array
#  \param[in] hessp hessian matrix of objective function applied on point p, returns vector array
#  \param[in] x0    initial value for optimization, vector array
#  \param[in] shape_solution shape of array of desired solution
#  \param[in] info_title determines which title is used to print information (optional)
#  \return data array of reconstruction
#  \return output of scipy.optimize.minimize function
def get_reconstruction(fun, jac, hessp, x0, shape_solution, info_title=False):
    iter_max = 100       # maximum number of iterations for solver
    tol = 1e-8          # tolerance for solver

    show_disp = True

    ## Provide bounds for optimization, i.e. intensities >= 0
    bounds = [[0,None]]*x0.size

    ## Start timing
    t0 = time.clock()

    ## Find approximate solution
    ## Look at http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    # res = minimize(method='Powell',     fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max})

    # res = minimize(method='L-BFGS-B',   fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac)
    # res = minimize(method='CG',         fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac)
    # res = minimize(method='BFGS',       fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac)

    ## Take incredibly long.
    # res = minimize(method='Newton-CG',  fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac, hessp=hessp)
    # res = minimize(method='trust-ncg',  fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max}, jac=jac, hessp=hessp)

    ## Find approximate solution using bounds
    res = minimize(method='L-BFGS-B',   fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max, 'disp': show_disp}, jac=jac, bounds=bounds)
    # res = minimize(method='TNC',   fun=fun, x0=x0, tol=tol, options={'maxiter': iter_max, 'disp': show_disp}, jac=jac, bounds=bounds)


    ## Stop timing
    time_elapsed = time.clock() - t0

    ## Print optimizer status
    if info_title is not None:
        print_status_optimizer(res, time_elapsed, info_title)

    return [res.x.reshape(shape_solution), res]


def plot_array(nda,title="no_title"):
    fig = plt.figure()
    
    plt.imshow(nda, cmap="Greys_r")
    plt.title(title)
    plt.show(block=False)


def show_reconstruction_results(original_nda, observed_nda, reconstruction_nda, reconstruction_TK1_nda=None):
    
    l2_error_obs = np.linalg.norm(observed_nda-original_nda)
    l2_error_TK0 = np.linalg.norm(reconstruction_nda-original_nda)

    fig = plt.figure()
    
    if reconstruction_TK1_nda is None:
        plt.subplot(221)
        plt.imshow(original_nda, cmap="Greys_r")
        plt.title("Original")

        plt.subplot(223)
        ax=plt.imshow(observed_nda, cmap="Greys_r")
        plt.title("Observed\n l2-error = "+ str(l2_error_obs))
        plt.colorbar(ax)

        plt.subplot(222)
        ax=plt.imshow(reconstruction_nda, cmap="Greys_r")
        plt.title("Reconstruction\n l2-error = "+ str(l2_error_TK0))
        plt.colorbar(ax)

        plt.subplot(224)
        ax=plt.imshow(abs(reconstruction_nda-original_nda), cmap="jet")
        plt.title("TK0: abs(recon-original) ")
        plt.colorbar(ax)

    else:
        l2_error_TK1 = np.linalg.norm(reconstruction_TK1_nda-original_nda)

        plt.subplot(231)
        ax=plt.imshow(original_nda, cmap="Greys_r")
        plt.title("Original")
        plt.colorbar(ax)

        plt.subplot(234)
        ax=plt.imshow(observed_nda, cmap="Greys_r")
        plt.title("Observed\n l2-error = "+ str(l2_error_obs))
        plt.colorbar(ax)

        plt.subplot(232)
        ax=plt.imshow(reconstruction_nda, cmap="Greys_r")
        plt.title("TK0 reconstruction\n l2-error = "+ str(l2_error_TK0))
        plt.colorbar(ax)

        plt.subplot(235)
        ax=plt.imshow(abs(reconstruction_nda-original_nda), cmap="jet")
        plt.title("TK0: abs(recon-original) ")
        plt.colorbar(ax)

        plt.subplot(233)
        ax=plt.imshow(reconstruction_TK1_nda, cmap="Greys_r")
        plt.title("TK1 reconstruction\n l2-error = "+ str(l2_error_TK1))
        plt.colorbar(ax)

        plt.subplot(236)
        ax=plt.imshow(abs(reconstruction_TK1_nda-original_nda), cmap="jet")
        plt.title("TK1: abs(recon-original) ")
        plt.colorbar(ax)

    plt.show(block=False)


## Print information stored in the result variable obtained from
#  scipy.optimize.minimize.
#  \param[in] res output from scipy.optimize.minimize
#  \param[in] time_elapsed measured time via time.clock() (optional)
#  \param[in] title title printed on the screen for subsequent information
def print_status_optimizer(res, time_elapsed=None, title="Overview"):
    print("Result optimization: %s" %(title))
    print("\t%s" % (res.message))
    print("\tCurrent value of objective function: %s" % (res.fun))
    try:
        print("\tIterations: %s" % (res.nit))
    except:
        None
    try:
        print("\tFunction evaluations: %s" % (res.nfev))
    except:
        None
    try:
        print("\tGradient evaluations: %s" % (res.njev))
    except:
        None
    try:
        print("\tHessian evaluations: %s" % (res.nhev))
    except:
        None

    if time_elapsed is not None:
        print("\tElapsed time for optimization: %s seconds" %(time_elapsed))


## Convolve image by using different boundary conditions to figure out
#  which adjoint operator reasonably fulfills |(Ax,y) - (x,A'y)| = 0
#  for the Gaussian blurring operation
#  \param[in] kernel used for Gaussian blurring
#  \param[in] image_sitk only defines the dimension which will be tested
def check_adjoint_Gaussian_blurring_boundary_conditions(kernel, image_sitk):

    dim = image_sitk.GetDimension()
    nda = sitk.GetArrayFromImage(image_sitk)

    if dim is 2:
        x = 255 * np.random.rand(nda.shape[0], nda.shape[1])
        y = 255 * np.random.rand(nda.shape[0], nda.shape[1])

    else:
        x = 255 * np.random.rand(nda.shape[0], nda.shape[1], nda.shape[2])
        y = 255 * np.random.rand(nda.shape[0], nda.shape[1], nda.shape[2])


    print("Gaussian Blurring: Test adjoint operator with various boundary conditions")

    ## Symmetric boundary conditions
    Ax = ndimage.convolve(x, kernel, mode='reflect')
    ATy = ndimage.convolve(y, kernel, mode='reflect')
    abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )
    print("\tSymmetric: |(Ax,y) - (x,A'y)| = %s" %(abs_diff))

    ## Periodic/Circular boundary conditions
    Ax = ndimage.convolve(x, kernel, mode='wrap')
    ATy = ndimage.convolve(y, kernel, mode='wrap')
    abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )
    print("\tPeriodic: |(Ax,y) - (x,A'y)| = %s (chosen for Gaussian blurring)" %(abs_diff))

    ## Mirror boundary conditions (?)
    Ax = ndimage.convolve(x, kernel, mode='mirror')
    ATy = ndimage.convolve(y, kernel, mode='mirror')
    abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )
    print("\tMirror: |(Ax,y) - (x,A'y)| = %s" %(abs_diff))

    ## Zero boundary conditions
    Ax = ndimage.convolve(x, kernel, mode='constant')
    ATy = ndimage.convolve(y, kernel, mode='constant')
    abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )
    print("\tZero: |(Ax,y) - (x,A'y)| = %s" %(abs_diff))


## Convolve image by using different boundary conditions to figure out
#  which adjoint operator reasonably fulfills |(Ax,y) - (x,A'y)| = 0
#  for the differential operation
#  \param[in] kernel_D kernel defining differentiation
#  \param[in] kernel_D_adj kernel defining adjoint operation to kernel_D
#  \param[in] image_sitk only defines the dimension which will be tested
def check_adjoint_differentiation_boundary_conditions(kernel_D, kernel_D_adj, image_sitk):

    dim = image_sitk.GetDimension()
    nda = sitk.GetArrayFromImage(image_sitk)

    ## Adjoint for forward difference is negative backward difference:
    # kernel = get_forward_difference_x_kernel(dim)
    # kernel_adjoint = - get_backward_difference_x_kernel(dim)

    ## Adjoint for forward difference is negative backward difference:
    # kernel = get_forward_difference_y_kernel(dim)
    # kernel_adjoint = - get_backward_difference_y_kernel(dim)    
    

    if dim is 2:
        x = 255 * np.random.rand(nda.shape[0], nda.shape[1])
        y = 255 * np.random.rand(nda.shape[0], nda.shape[1])

    else:
        x = 255 * np.random.rand(nda.shape[0], nda.shape[1], nda.shape[2])
        y = 255 * np.random.rand(nda.shape[0], nda.shape[1], nda.shape[2])


    print("Differentiation: Test adjoint operator with various boundary conditions")

    ## Symmetric boundary conditions
    Ax = ndimage.convolve(x, kernel_D, mode='reflect')
    ATy = ndimage.convolve(y, kernel_D_adj, mode='reflect')
    abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )
    print("\tSymmetric: |(Ax,y) - (x,A'y)| = %s" %(abs_diff))

    ## Periodic/Circular boundary conditions
    Ax = ndimage.convolve(x, kernel_D, mode='wrap')
    ATy = ndimage.convolve(y, kernel_D_adj, mode='wrap')
    abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )
    print("\tPeriodic: |(Ax,y) - (x,A'y)| = %s (chosen for differential convolution)" %(abs_diff))

    ## Mirror boundary conditions (?)
    Ax = ndimage.convolve(x, kernel_D, mode='mirror')
    ATy = ndimage.convolve(y, kernel_D_adj, mode='mirror')
    abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )
    print("\tMirror: |(Ax,y) - (x,A'y)| = %s" %(abs_diff))

    ## Zero boundary conditions
    Ax = ndimage.convolve(x, kernel_D, mode='constant')
    ATy = ndimage.convolve(y, kernel_D_adj, mode='constant')
    abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )
    print("\tZero: |(Ax,y) - (x,A'y)| = %s" %(abs_diff))


"""
Unit Test Class
"""
class TestUM(unittest.TestCase):

    accuracy = 4
    dir_input = "data/"
    dir_output = "results/"


    def setUp(self):
        pass


    ## Test adjoint Gaussian blurring operation in 2D, 
    #  i.e. check |(Ax,y) - (x,A'y)| = 0
    def test_01_AdjointBlurringOperation_2D(self):

        shape = (100,200)

        alpha_cut = 3
        Cov_2D = np.zeros((2,2))
        Cov_2D[0,0] = 2
        Cov_2D[1,1] = 2


        kernel = get_gaussian_kernel(Cov_2D, alpha_cut);

        x = 255 * np.random.rand(shape[0],shape[1])
        y = 255 * np.random.rand(shape[0],shape[1])


        # else:
        #     x = 255 * np.random.rand(nda.shape[0], nda.shape[1], nda.shape[2])
        #     y = 255 * np.random.rand(nda.shape[0], nda.shape[1], nda.shape[2])

        ## Symmetric boundary conditions
        Ax = getA(x,kernel)
        ATy = getAT(y,kernel)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )

    ## Test adjoint Gaussian blurring operation in 3D, 
    #  i.e. check |(Ax,y) - (x,A'y)| = 0
    def test_01_AdjointBlurringOperation_3D(self):

        shape = (50,60,70)

        alpha_cut = 3

        Cov_3D = np.zeros((3,3))
        Cov_3D[0,0] = 2
        Cov_3D[1,1] = 2
        Cov_3D[2,2] = 2

        kernel = get_gaussian_kernel(Cov_3D, alpha_cut);

        x = 255 * np.random.rand(shape[0],shape[1],shape[2])
        y = 255 * np.random.rand(shape[0],shape[1],shape[2])


        ## Symmetric boundary conditions
        Ax = getA(x,kernel)
        ATy = getAT(y,kernel)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy-1), 0 )


    ## Test adjoint forward difference operation in 2D, 
    #  i.e. check |(Ax,y) - (x,A'y)| = 0
    def test_02_AdjointForwardDifferenceOperation_2D(self):

        dim = 2
        shape = (50,60)

        ## 1) Check forward difference in x-direction
        kernel_D = get_forward_difference_x_kernel(dim);
        kernel_D_adj = -get_backward_difference_x_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1])
        y = 255 * np.random.rand(shape[0],shape[1])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


        ## 2) Check forward difference in y-direction
        kernel_D = get_forward_difference_y_kernel(dim);
        kernel_D_adj = -get_backward_difference_y_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1])
        y = 255 * np.random.rand(shape[0],shape[1])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


    ## Test adjoint forward difference operation in 3D, 
    #  i.e. check |(Ax,y) - (x,A'y)| = 0
    def test_02_AdjointForwardDifferenceOperation_3D(self):

        dim = 3
        shape = (50,60,70)

        ## 1) Check forward difference in x-direction
        kernel_D = get_forward_difference_x_kernel(dim);
        kernel_D_adj = -get_backward_difference_x_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1],shape[2])
        y = 255 * np.random.rand(shape[0],shape[1],shape[2])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


        ## 2) Check forward difference in y-direction
        kernel_D = get_forward_difference_y_kernel(dim);
        kernel_D_adj = -get_backward_difference_y_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1],shape[2])
        y = 255 * np.random.rand(shape[0],shape[1],shape[2])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


        ## 3) Check forward difference in z-direction
        kernel_D = get_forward_difference_z_kernel(dim);
        kernel_D_adj = -get_backward_difference_z_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1],shape[2])
        y = 255 * np.random.rand(shape[0],shape[1],shape[2])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


    ## Test adjoint backward difference operation in 2D, 
    #  i.e. check |(Ax,y) - (x,A'y)| = 0
    def test_03_AdjointBackwardDifferenceOperation_2D(self):

        dim = 2
        shape = (50,60)

        ## 1) Check backward difference in x-direction
        kernel_D = get_backward_difference_x_kernel(dim);
        kernel_D_adj = -get_forward_difference_x_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1])
        y = 255 * np.random.rand(shape[0],shape[1])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


        ## 2) Check backward difference in y-direction
        kernel_D = get_backward_difference_y_kernel(dim);
        kernel_D_adj = -get_forward_difference_y_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1])
        y = 255 * np.random.rand(shape[0],shape[1])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


    ## Test adjoint forward difference operation in 3D, 
    #  i.e. check |(Ax,y) - (x,A'y)| = 0
    def test_03_AdjointBackwardDifferenceOperation_3D(self):

        dim = 3
        shape = (50,60,70)

        ## 1) Check forward difference in x-direction
        kernel_D = get_backward_difference_x_kernel(dim);
        kernel_D_adj = -get_forward_difference_x_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1],shape[2])
        y = 255 * np.random.rand(shape[0],shape[1],shape[2])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


        ## 2) Check forward difference in y-direction
        kernel_D = get_backward_difference_y_kernel(dim);
        kernel_D_adj = -get_forward_difference_y_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1],shape[2])
        y = 255 * np.random.rand(shape[0],shape[1],shape[2])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


        ## 3) Check forward difference in z-direction
        kernel_D = get_backward_difference_z_kernel(dim);
        kernel_D_adj = -get_forward_difference_z_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1],shape[2])
        y = 255 * np.random.rand(shape[0],shape[1],shape[2])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Ax,y) - (x,A'y)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


    ## Test Laplacian operation in 2D, 
    #  i.e. check L(x) = (Dfx - Dbx + Dfy - Dby)(x)
    def test_04_LaplacianOperation_2D(self):

        dim = 2
        shape = (50,60)

        kernel_L = get_laplacian_kernel(dim)

        kernel_Dfx = get_forward_difference_x_kernel(dim)
        kernel_Dbx = get_backward_difference_x_kernel(dim)

        kernel_Dfy = get_forward_difference_y_kernel(dim)
        kernel_Dby = get_backward_difference_y_kernel(dim)

        x = 255 * np.random.rand(shape[0],shape[1])

        ## Compute all variables involved
        L_x = getDx(x,kernel_L)
        Dfx_x = getDx(x,kernel_Dfx)
        Dbx_x = getDx(x,kernel_Dbx)
        Dfy_x = getDx(x,kernel_Dfy)
        Dby_x = getDx(x,kernel_Dby)

        ## Check L(x) = (Dfx - Dbx + Dfy - Dby)(x)
        norm_diff = np.linalg.norm( L_x - ( Dfx_x - Dbx_x + Dfy_x - Dby_x) )
        self.assertEqual(np.around(
            norm_diff
            , decimals = self.accuracy), 0 )


    ## Test Laplacian operation in 3D, 
    #  i.e. check L(x) = (Dfx - Dbx + Dfy - Dby + Dfz - Dbz)(x)
    def test_04_LaplacianOperation_3D(self):

        dim = 3
        shape = (50,60,70)

        kernel_L = get_laplacian_kernel(dim)

        kernel_Dfx = get_forward_difference_x_kernel(dim)
        kernel_Dbx = get_backward_difference_x_kernel(dim)

        kernel_Dfy = get_forward_difference_y_kernel(dim)
        kernel_Dby = get_backward_difference_y_kernel(dim)

        kernel_Dfz = get_forward_difference_z_kernel(dim)
        kernel_Dbz = get_backward_difference_z_kernel(dim)

        x = 255 * np.random.rand(shape[0],shape[1],shape[2])

        ## Compute all variables involved
        L_x = getDx(x,kernel_L)
        Dfx_x = getDx(x,kernel_Dfx)
        Dbx_x = getDx(x,kernel_Dbx)
        Dfy_x = getDx(x,kernel_Dfy)
        Dby_x = getDx(x,kernel_Dby)
        Dfz_x = getDx(x,kernel_Dfz)
        Dbz_x = getDx(x,kernel_Dbz)

        ## Check L(x) = (Dfx - Dbx + Dfy - Dby + Dfz - Dbz)(x)
        norm_diff = np.linalg.norm( L_x - (Dfx_x - Dbx_x + Dfy_x - Dby_x + Dfz_x - Dbz_x) )
        self.assertEqual(np.around(
            norm_diff
            , decimals = self.accuracy), 0 )


    ## Test adjoint Laplacian operation is self-adjoint in 2D, 
    #  i.e. check |(Lx,y) - (x,Ly)| = 0
    def test_04_AdjointLaplacianOperation_2D(self):

        dim = 2
        shape = (50,60)

        kernel_D = get_laplacian_kernel(dim);
        kernel_D_adj = get_laplacian_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1])
        y = 255 * np.random.rand(shape[0],shape[1])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check|(Lx,y) - (x,Ly)| = 0 
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


    ## Test adjoint Laplacian operation is self-adjoint in 3D, 
    #  i.e. check |(Lx,y) - (x,Ly)| = 0
    def test_04_AdjointLaplacianOperation_3D(self):

        dim = 3
        shape = (50,60,70)

        kernel_D = get_laplacian_kernel(dim);
        kernel_D_adj = get_laplacian_kernel(dim);

        x = 255 * np.random.rand(shape[0],shape[1],shape[2])
        y = 255 * np.random.rand(shape[0],shape[1],shape[2])

        ## Symmetric boundary conditions
        Ax = getDx(x, kernel_D)
        ATy = getDTx(y, kernel_D_adj)
        abs_diff = np.abs( np.sum(Ax*y) - np.sum(ATy*x) )

        ## Check |(Lx,y) - (x,Ly)| = 0
        self.assertEqual(np.around(
            abs_diff
            , decimals = self.accuracy), 0 )


"""
Main Function
"""
if __name__ == '__main__':

    dir_input = "data/"
    dir_output = "results/"
    filename_HR_volume = "FetalBrain_reconstruction_4stacks"
    filename_stack = "FetalBrain_stack2_registered"
    filename_slice = "FetalBrain_stack2_registered_midslice"

    # filename_2D = "2D_BrainWeb"
    # filename_2D = "2D_SingleDot_50"
    # filename_2D = "2D_Cross_50"
    # filename_2D = "2D_Text"
    # filename_2D = "2D_Cameraman_256"
    # filename_2D = "2D_House_256"
    filename_2D = "2D_SheppLoganPhantom_512"
    # filename_2D = "2D_Lena_512"
    # filename_2D = "2D_Boat_512"
    # filename_2D = "2D_Man_1024"

    # filename_3D = "FetalBrain_reconstruction_4stacks"
    # filename_3D = "3D_SingleDot_50"
    # filename_3D = "3D_Cross_50"
    filename_3D = "3D_SheppLoganPhantom_64"
    # filename_3D = "3D_SheppLoganPhantom_128"


    ## Choose set-up used in here
    # filename = filename_2D
    filename = filename_3D

    noise_level = 0                 ## Noise level for test image, default = 0.01
    alpha_cut = 3                   ## Cut-off distance alpha_cut*sigma for gaussian kernel computation
    alpha = 0                      ## Regularization parameter
    
    use_SPD_formulation = False


    if alpha is not 0:
        use_SPD_formulation = True
    
    Cov_2D = np.zeros((2,2))
    Cov_2D[0,0] = 9
    Cov_2D[1,1] = 9
    # Sigma_2D = np.sqrt(Cov_2D.diagonal())

    Cov_3D = np.zeros((3,3))
    # Cov_3D[0,0] = 0.26786367
    # Cov_3D[1,1] = 0.26786367
    # Cov_3D[2,2] = 2.67304559
    Cov_3D[0,0] = 2
    Cov_3D[1,1] = 2
    Cov_3D[2,2] = 2
    # Sigma_3D = np.sqrt(Cov_3D.diagonal())


    ## Read images
    HR_volume_sitk = sitk.ReadImage(dir_input + filename_HR_volume + ".nii.gz", sitk.sitkFloat32)
    stack_sitk = sitk.ReadImage(dir_input + filename_stack + ".nii.gz", sitk.sitkFloat32)
    slice_sitk = sitk.ReadImage(dir_input + filename_slice + ".nii.gz", sitk.sitkFloat32)

    image_sitk = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat32)
    

    ## Define covariance matrix depending on dimensionality
    dim = image_sitk.GetDimension()
    if dim is 2:
        Cov = Cov_2D
    else:
        Cov = Cov_3D

    ## Get smoothing kernel
    kernel = get_gaussian_kernel(Cov, alpha_cut);

    ## Simulate observed image
    image_observed_sitk = get_blurred_noisy_image(image_sitk, Cov, alpha_cut, noise_level, multiplicative_noise=False)

    original_nda = sitk.GetArrayFromImage(image_sitk)
    observed_nda = sitk.GetArrayFromImage(image_observed_sitk)
    shape = observed_nda.shape

    # sitkh.show_sitk_image(image_sitk=image_sitk, overlay_sitk=image_observed_sitk)

    kernel_Dx = get_forward_difference_x_kernel(dim)
    kernel_Dx_adj = -get_backward_difference_x_kernel(dim)

    kernel_Dy = get_forward_difference_y_kernel(dim)
    kernel_Dy_adj = -get_backward_difference_y_kernel(dim)

    if dim is 2:
        kernel_Dz = None
        kernel_Dz_adj = None
    else:
        kernel_Dz = get_forward_difference_z_kernel(dim)
        kernel_Dz_adj = -get_backward_difference_z_kernel(dim)

    # plot_array(nda_dx)

    g = observed_nda
    ATg = getAT(g,kernel)

    ## Compute ATAATg = A'AA'g
    ATAATg = getATA(ATg, kernel) 

    ## Compute DTDATg = D'DA'g
    DTDATg = getDTDx(ATg, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)
    
    ## op0_ATg = Compute (A'A + alpha)(A'g)
    op0_ATg = ATAATg + alpha*ATg

    ## op1_ATg = Compute (A'A + alpha*D'D)(A'g)
    op1_ATg = ATAATg + alpha*DTDATg



    f_TK0 = lambda x: TK0(x, g.flatten(), kernel, alpha, shape)
    f_TK0_grad = lambda x: TK0_grad(x, ATg.flatten(), kernel, alpha, shape)
    f_TK0_hess_p = lambda x, p: TK0_hess_p(x, p, kernel, alpha, shape)

    f_TK0_SPD = lambda x: TK0_SPD(x, ATg.flatten(), kernel, alpha, shape)
    f_TK0_SPD_grad = lambda x: TK0_SPD_grad(x, (op0_ATg).flatten(), kernel, alpha, shape)
    f_TK0_SPD_hess_p = lambda x, p: TK0_SPD_hess_p(x, p, kernel, alpha, shape)


    f_TK1 = lambda x: TK1(x, g.flatten(), kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dz)
    f_TK1_grad = lambda x: TK1_grad(x, ATg.flatten(), kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)
    f_TK1_hess_p = lambda x, p: TK1_hess_p(x, p, kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)

    f_TK1_SPD = lambda x: TK1_SPD(x, ATg.flatten(), kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)
    f_TK1_SPD_grad = lambda x: TK1_SPD_grad(x, op1_ATg.flatten(), kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)
    f_TK1_SPD_hess_p = lambda x, p: TK1_SPD_hess_p(x, p, kernel, alpha, shape, kernel_Dx, kernel_Dy, kernel_Dx_adj, kernel_Dy_adj, kernel_Dz, kernel_Dz_adj)

    
    f_augmented = lambda x: TK0_augmented(x, g.flatten(), kernel, alpha, shape)

    if use_SPD_formulation:
        f_TK0 = f_TK0_SPD
        f_TK0_grad = f_TK0_SPD_grad
        f_TK0_hess_p = f_TK0_SPD_hess_p

        f_TK1 = f_TK1_SPD
        f_TK1_grad = f_TK1_SPD_grad
        f_TK1_hess_p = f_TK1_SPD_hess_p

    if dim is 3:
        sitkh.show_sitk_image(image_sitk=image_sitk, overlay_sitk=image_observed_sitk, title=filename+"_original+observed")

    ## Compute TK0 solution
    [reconstruction_TK0_nda, res_minimizer_TK0] \
        = get_reconstruction(fun=f_TK0, jac=f_TK0_grad, hessp=f_TK0_hess_p, x0=ATg.flatten(), shape_solution=ATg.shape, info_title="TK0")

    image_reconstructed_TK0_sitk = sitk.GetImageFromArray(reconstruction_TK0_nda)
    image_reconstructed_TK0_sitk.CopyInformation(image_sitk)

    if dim is 3:
        sitkh.show_sitk_image(image_sitk=image_sitk, overlay_sitk=image_reconstructed_TK0_sitk, title=filename+"_original+TK0_recon_alpha"+str(alpha))
        # sitkh.show_sitk_image(image_sitk=image_observed_sitk, overlay_sitk=image_reconstructed_TK0_sitk, title=filename+"_TK0_observed+recon")
        

    ## Compute TK1 solution
    [reconstruction_TK1_nda, res_minimizer_TK1] \
        = get_reconstruction(fun=f_TK1, jac=f_TK1_grad, hessp=f_TK1_hess_p, x0=ATg.flatten(), shape_solution=ATg.shape, info_title="TK1")

    image_reconstructed_TK1_sitk = sitk.GetImageFromArray(reconstruction_TK1_nda)
    image_reconstructed_TK1_sitk.CopyInformation(image_sitk)

    if dim is 3:
        sitkh.show_sitk_image(image_sitk=image_sitk, overlay_sitk=image_reconstructed_TK1_sitk, title=filename+"_original+TK1_recon_alpha"+str(alpha))
        # sitkh.show_sitk_image(image_sitk=image_observed_sitk, overlay_sitk=image_reconstructed_TK1_sitk, title=filename+"_TK1_observed+recon")
    

    if dim is 2:
        # show_reconstruction_results(original_nda=original_nda, observed_nda=observed_nda, reconstruction_nda=reconstruction_TK0_nda)
        show_reconstruction_results(original_nda=original_nda, observed_nda=observed_nda, reconstruction_nda=reconstruction_TK0_nda, reconstruction_TK1_nda=reconstruction_TK1_nda)


    

    
    # res = minimize(fun=f_TK0, jac=f_TK0_grad, hessp=f_TK0_hess_p, x0=ATg.flatten(), method='trust-ncg', tol=tol)
    # res = leastsq(func=f_augmented, x0=ATg.flatten(), xtol=tol, maxfev=1000) #crashes
    # image_reconstructed_TK0_sitk = sitk.GetImageFromArray(res[0].reshape(g.shape))

    # else:
    #     ## Get smoothing kernel
    #     kernel = get_gaussian_kernel(Cov_3D, alpha_cut);

    #     ## Simulate observed image
    #     # image_observed_3D_sitk = get_blurred_noisy_image(image_3D_sitk, Cov_3D, alpha_cut, multiplicative_noise=True)

    #     # sitkh.show_sitk_image(image_sitk=image_3D_sitk, overlay_sitk=image_observed_3D_sitk)

    #     # kernel_dx = get_forward_difference_x_kernel(dim)
    #     # kernel_dx = get_forward_difference_y_kernel(dim)
    #     kernel_dx = get_forward_difference_z_kernel(dim)

    #     nda = sitk.GetArrayFromImage(image_3D_sitk)
    #     observed_nda = sitk.GetArrayFromImage(image_observed_3D_sitk)
    #     shape = observed_nda.shape

    #     nda_dx = getDx(nda, kernel_dx)
    #     test = sitk.GetImageFromArray(nda_dx)
    #     test.CopyInformation(image_3D_sitk)

    #     sitkh.show_sitk_image(image_sitk=image_3D_sitk, overlay_sitk=test)


    plt.draw()

    # check_adjoint_Gaussian_blurring_boundary_conditions(kernel, image_sitk)
    # check_adjoint_differentiation_boundary_conditions(kernel_Dx, kernel_Dx_adj, image_sitk)

    """
    Unit tests:
    """
    print("\nUnit tests:\n--------------")
    unittest.main()
