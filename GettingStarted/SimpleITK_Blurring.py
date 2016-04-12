## \file SimpleITK_Blurring.py
#  \brief  Computation of own Gaussian Kernels in 3D
#  \note Code tested against sitk.SmoothingRecursiveGaussianImageFilter 
#       \par 2D images: satisfying results also for tested arbitrary spacings
#       \par 3D images: satisfying results only for uniform spacings (cf unit tests)
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015

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

## Check whether variance-covariance matrix is SPD
#  \param Sigma matrix to check
def is_SPD(Sigma):
    tol = 1e-8
    
    ## is positive definite
    bool_PD = np.all(np.linalg.eigvals(Sigma)>0)

    ## is symmetric
    bool_Sym = np.linalg.norm(Sigma.transpose() - Sigma) < tol

    return bool_PD and bool_Sym


## Obtain value of ellipsoidal function (2 or 3 dimensional)
#  \param points to check given as array \in \R^{(2 or 3)\times N}
#  \param origin origin of ellipsoid \in \R^{(2 or 3)}
#  \param Sigma variance-covariance matrix defining the ellipsoid
#  \return value of ellipsoidal function 
def evaluate_function_ellipsoid(points, origin, Sigma):

    points = np.array(points)
    origin = np.array(origin)

    try:
        ## Guarantee that origin is column vector, i.e. in \R^{(2 or 3)}
        if origin.size is 2 or origin.size is 3:
            origin = origin.reshape(origin.size,-1)

        else:
            raise ValueError("Error: origin is not in R^2 or R^3")


        ## If only one point is given: Reshape to column vector
        if points.size is 2 or points.size is 3:
            points = points.reshape(points.size,-1)

        ## Check whether input parameters are well-defined
        if (points.shape[0] is not 2 and points.shape[0] is not 3) \
            or (points.shape[0] is not origin.size) \
            or (Sigma.shape[0] is not Sigma.shape[1]) \
            or (points.shape[0] is not Sigma.shape[0]):
            raise ValueError("Error: Parameters must be of dimension 2 or 3")

        ## Check whether variance-covariance matrix is SPD
        elif not is_SPD(Sigma):
            raise ValueError("Error: Sigma is not SPD")

        else:
            ## Compute value according to equation of ellipsoid
            value = np.sum((points-origin)*np.linalg.inv(Sigma).dot(points-origin), 0)
            return value

    except ValueError as err:
        print(err.args[0])


## Compute multivariate Gaussian 
#  \param points to check given as array \in \R^{dim \times N}
#  \param mu mean of gaussian \in \R^{dim}
#  \param Sigma variance-covariance matrix \in \R^{dim \times dim}
#  \return values of (multivariate) Gaussian distribution
def compute_gaussian(points, mu, Sigma):

    points = np.array(points)
    mu = np.array(mu)

    dim = Sigma.shape[0]

    try:
        ## Guarantee that mu is column vector, i.e. in \R^{(2 or 3)}
        if mu.size is 2 or mu.size is 3:
            mu = mu.reshape(mu.size,-1)

        else:
            raise ValueError("Error: mu is not in R^2 or R^3")


        ## If only one point is given: Reshape to column vector
        if points.size is 2 or points.size is 3:
            points = points.reshape(points.size,-1)

        ## Check whether input parameters are well-defined
        if (points.shape[0] is not 2 and points.shape[0] is not 3) \
            or (points.shape[0] is not mu.size) \
            or (Sigma.shape[0] is not Sigma.shape[1]) \
            or (points.shape[0] is not Sigma.shape[0]):
            raise ValueError("Error: Parameters must be of dimension 2 or 3")

        ## Check whether variance-covariance matrix is SPD
        elif not is_SPD(Sigma):
            raise ValueError("Error: Sigma is not SPD")

        else:
            ## Compute value according to equation of ellipsoid
            value = np.sum((points-mu)*np.linalg.inv(Sigma).dot(points-mu), 0)
            value = np.exp(-0.5*value)/np.sqrt((2*np.pi)**dim * np.linalg.det(Sigma)) 
            
            return value

    except ValueError as err:
        print(err.args[0])


## Determine whether points are within ellipsoid (2 or 3 dimensional)
#  \param points to check given as array \in \R^{(2 or 3)\times N}
#  \param origin origin of ellipsoid  \in \R^{(2 or 3)}
#  \param Sigma variance-covariance matrix defining the ellipsoid
#  \param cutoff_level
#  \return boolean array stating whether points are in ellipsoid  and corrsponding values
def is_in_ellipsoid(points, origin, Sigma, cutoff_level):
    eps = 1e-8          ## numerical tolerance for comparison

    values = evaluate_function_ellipsoid(points, origin, Sigma)
    # print values

    ## Points are within ellipsoid
    return values <= cutoff_level+eps, values


## Scale axis of ellipsoid defined by the variance covariance matrix
#  \param Sigma variance covariance matrix
#  \param scale factor to multiply with main axis lenghts
#  \return scaled variance covariance matrix
def get_scaled_variance_covariance_matrix(Sigma, scales):

    ## Perform SVD
    # U,s,V = np.linalg.svd(Sigma)

    ## Scale variances
    # print("Variances before scaling with factor(s)=%s: %s" %(scales,s))
    # s = scales*s;
    # print("Variances after scaling with factor(s)=%s: %s" %(scales,s))

    ## Computed scaled variance covariance matrix
    # Sigma = U.dot(np.diag(s)).dot(np.transpose(V))

    # return Sigma

    # Scaling = np.diag(scales)
    # Scaling = np.diag(1/scales)

    # Sigma =  Scaling.dot(Sigma).dot(Scaling)
    # print Sigma
    return Sigma
    # return Scaling.dot(Sigma).dot(Scaling_inv)


## get smoothing kernel
#  \param image_sitk
#  \param Sigma
#  \param origin
#  \param cutoff either
#  \return kernel 
#  \return reference
def get_smoothing_kernel(image_sitk, Sigma, origin, cutoff):

    dim = Sigma.shape[0]

    ## Convert to arrays
    cutoff = np.array(cutoff)
    spacing = np.array(image_sitk.GetSpacing())

    ## Scale to image space with cutoff-sigma
    scaling = spacing

    Sigma_image = get_scaled_variance_covariance_matrix(Sigma, scaling)
    
    ## Perform SVD
    U,s_image,V = np.linalg.svd(Sigma_image)

    # print s_image
    # U,s,V = np.linalg.svd(Sigma)
    # print s

    ## cutoff = cutoff_level, i.e. a scalar to determine cutoff level of ellipse
    if cutoff.size == 1:

        ## Maximumg length of vector in image space
        ## (Length of) vector representing the sphere such that cutoff-ellipse is for sure covered
        l_max_image = np.sqrt(np.linalg.norm(s_image)*cutoff)
        print("l_max_image = %s" %(l_max_image))
        l_max_image = np.ceil(l_max_image)


        if dim == 2:
            ## Generate intervals for x and y based on l_max_image
            step = 1
            x_interval = np.arange(-l_max_image, l_max_image+step,step)
            y_interval = np.arange(-l_max_image, l_max_image+step,step)

            ## Store reference/center/refpoint of kernel
            reference = np.array([len(x_interval), len(y_interval)])/2
            # print ("reference = %s" %reference)

            ## Generate arrays of 2D points
            X,Y = np.meshgrid(x_interval, y_interval, indexing='ij')    # 'ij' yields vertical x-coordinate for image!
            S = np.diag(spacing)
            points = S.dot(np.array([X.flatten(), Y.flatten()]))
            # points = np.array([X.flatten(), Y.flatten()])

            ## Determine points in ellipsoid
            bool, vals = is_in_ellipsoid(points, origin, Sigma_image, cutoff)

            ## Compute Gaussian values
            vals = np.exp(-0.5*vals)/np.sqrt((2*np.pi)**dim * np.linalg.det(Sigma_image)) 

            ## Reshape to grid defined by X and Y
            Bool = bool.reshape(x_interval.size, y_interval.size) 
            Vals = vals.reshape(x_interval.size, y_interval.size) 

            # print("%s-Bool = \n%s" %(Bool.shape,Bool))
            # print("%s-X = \n%s" %(X.shape,X))
            # print("%s-Y = \n%s" %(Y.shape,Y))
            # print("%s-Vals = \n%s" %(Vals.shape,Vals))

            ## Normalize values of kernel
            Phi_all = np.sum(Vals)*step**dim

            ## Set values out of desired ellipse to zero
            Vals[Bool==0] = 0

            ## Compute coverage of gaussian (i.e. approximation of confidence interval)
            Phi = np.sum(Vals[Bool])*step**dim

            ## Normalize values to kernel
            kernel = Vals/np.sum(Vals[Bool])

            ## Find rows which only contain zeros
            delete_ind_rows = []
            for i in range(0, kernel.shape[0]):
                if np.sum(kernel[i,:]) == 0:
                    # print("delete row %s" %i)
                    delete_ind_rows.append(i)

                    ## Update center/reference/mitpoint
                    if i<x_interval.size/float(2):
                        # print("reference[0]=%s" %reference[0])
                        reference[0] = reference[0]-1

            ## Find cols which only contain zeros
            delete_ind_cols = []
            for i in range(0, kernel.shape[1]):
                if np.sum(kernel[:,i]) == 0:
                    # print("delete col %s" %i)
                    delete_ind_cols.append(i)

                    ## Update center/reference/mitpoint
                    if i<y_interval.size/float(2):
                        reference[1] = reference[1]-1

            ## Delete rows and columns which only contain zeros
            X = np.delete(X, delete_ind_rows, axis=0)
            Y = np.delete(Y, delete_ind_rows, axis=0)
            kernel = np.delete(kernel, delete_ind_rows, axis=0)
            Bool = np.delete(Bool, delete_ind_rows, axis=0)

            X = np.delete(X, delete_ind_cols, axis=1)
            Y = np.delete(Y, delete_ind_cols, axis=1)
            kernel = np.delete(kernel, delete_ind_cols, axis=1)
            Bool = np.delete(Bool, delete_ind_cols, axis=1)

            # print("cutoff = %s" %cutoff)
            print("%s-Bool = \n%s" %(Bool.shape,Bool))
            print("%s-X = \n%s" %(X.shape,X))
            print("%s-Y = \n%s" %(Y.shape,Y))


        elif dim == 3:
            ## Generate intervals for x, y and z based on l_max_image
            step = 1
            x_interval = np.arange(-l_max_image, l_max_image+step,step)
            y_interval = np.arange(-l_max_image, l_max_image+step,step)
            z_interval = np.arange(-l_max_image, l_max_image+step,step)

            ## Store reference/center/refpoint of kernel
            reference = np.array([len(x_interval), len(y_interval), len(z_interval)])/2

            ## Generate arrays of 3D points
            X,Y,Z = np.meshgrid(x_interval, y_interval, z_interval, indexing='ij')    # 'ij' yields vertical x-coordinate for image!
            S = np.diag(spacing)
            points = S.dot(np.array([X.flatten(), Y.flatten(), Z.flatten()]))
            # points = np.array([X.flatten(), Y.flatten(), Z.flatten()])

            ## Determine points in ellipsoid
            bool, vals = is_in_ellipsoid(points, origin, Sigma_image, cutoff)

            ## Compute Gaussian values
            vals = np.exp(-0.5*vals)/np.sqrt((2*np.pi)**dim * np.linalg.det(Sigma_image)) 

            ## Reshape to grid defined by X, Y and Z
            Bool = bool.reshape(x_interval.size, y_interval.size, z_interval.size) 
            Vals = vals.reshape(x_interval.size, y_interval.size, z_interval.size) 

            # print("%s-Bool = \n%s" %(Bool.shape,Bool))
            # print("%s-X = \n%s" %(X.shape,X))
            # print("%s-Y = \n%s" %(Y.shape,Y))
            # print("%s-Z = \n%s" %(Z.shape,Z))
            # print("%s-Vals = \n%s" %(Vals.shape,Vals))

            ## Normalize values of kernel (rectangu)
            Phi_all = np.sum(Vals)*step**dim

            ## Set values out of desired ellipse to zero
            Vals[Bool==0] = 0

            ## Compute coverage of gaussian (i.e. approximation of confidence interval)
            Phi = np.sum(Vals[Bool])*step**dim

            ## Normalize values to kernel
            kernel = Vals/np.sum(Vals[Bool])

            ## Find x-planes which only contain zeros
            delete_ind_x = []
            for x in range(0, kernel.shape[0]):
                if np.sum(kernel[x,:,:]) == 0:
                    # print("delete x-plane %s" %x)
                    delete_ind_x.append(x)

                    ## Update center/reference/mitpoint
                    if x<x_interval.size/float(2):
                        # print("reference[0]=%s" %reference[0])
                        reference[0] = reference[0]-1

            ## Find y-planes which only contain zeros
            delete_ind_y = []
            for y in range(0, kernel.shape[1]):
                if np.sum(kernel[:,y,:]) == 0:
                    # print("delete y-plane %s" %y)
                    delete_ind_y.append(y)

                    ## Update center/reference/mitpoint
                    if y<y_interval.size/float(2):
                        # print("reference[0]=%s" %reference[0])
                        reference[1] = reference[1]-1

            ## Find z-planes which only contain zeros
            delete_ind_z = []
            for z in range(0, kernel.shape[2]):
                if np.sum(kernel[:,:,z]) == 0:
                    # print("delete z-plane %s" %z)
                    delete_ind_z.append(z)

                    ## Update center/reference/mitpoint
                    if z<z_interval.size/float(2):
                        # print("reference[0]=%s" %reference[0])
                        reference[2] = reference[2]-1


            ## Delete all plains which only contain zeros
            X = np.delete(X, delete_ind_x, axis=0)
            Y = np.delete(Y, delete_ind_x, axis=0)
            Z = np.delete(Z, delete_ind_x, axis=0)
            kernel = np.delete(kernel, delete_ind_x, axis=0)
            Bool = np.delete(Bool, delete_ind_x, axis=0)

            X = np.delete(X, delete_ind_y, axis=1)
            Y = np.delete(Y, delete_ind_y, axis=1)
            Z = np.delete(Z, delete_ind_y, axis=1)
            kernel = np.delete(kernel, delete_ind_y, axis=1)
            Bool = np.delete(Bool, delete_ind_y, axis=1)

            X = np.delete(X, delete_ind_z, axis=2)
            Y = np.delete(Y, delete_ind_z, axis=2)
            Z = np.delete(Z, delete_ind_z, axis=2)
            kernel = np.delete(kernel, delete_ind_z, axis=2)
            Bool = np.delete(Bool, delete_ind_z, axis=2)

            # print("cutoff = %s" %cutoff)
            # print("%s-Bool = \n%s" %(Bool.shape,Bool))
            # print("%s-X = \n%s" %(X.shape,X))
            # print("%s-Y = \n%s" %(Y.shape,Y))
            # print("%s-Z = \n%s" %(Z.shape,Z))


        else:
            raise ValueError("Error: Dimension must be 2 or 3")



    ## cutoff represents kernel size
    else:
        if dim == 2:
            ## create intervalls
            step = 1
            x_interval = np.arange(0,cutoff[0],step)
            y_interval = np.arange(0,cutoff[1],step)

            ## symmetry around zero
            x_interval = x_interval-x_interval.mean()
            y_interval = y_interval-y_interval.mean()

            ## Store reference/center/refpoint of kernel
            # origin = reference = np.array(kernel.shape)/2
            reference = np.array([len(x_interval), len(y_interval)])/2
            # print ("reference = %s" %reference)

            ## Generate arrays of 2D points
            X,Y = np.meshgrid(x_interval, y_interval, indexing='ij')    # 'ij' yields vertical x-coordinate for image!
            points = np.array([X.flatten(), Y.flatten()])

            ## Compute Gaussian values
            vals = compute_gaussian(points, origin, Sigma)

            ## Reshape to grid defined by X and Y
            Vals = vals.reshape(x_interval.size,-1)

            ## Compute coverage of gaussian (i.e. approximation of confidence interval)
            Phi = np.sum(Vals)*step**dim

            ## Normalize values to kernel
            kernel = Vals/np.sum(vals)

        elif dim == 3:
            ## create intervalls
            step = 1
            x_interval = np.arange(0,cutoff[0],step)
            y_interval = np.arange(0,cutoff[1],step)
            z_interval = np.arange(0,cutoff[1],step)

            ## symmetry around zero
            x_interval = x_interval-x_interval.mean()
            y_interval = y_interval-y_interval.mean()
            z_interval = z_interval-z_interval.mean()
            
            ## Store reference/center/refpoint of kernel
            # origin = reference = np.array(kernel.shape)/2
            reference = np.array([len(x_interval), len(y_interval), len(z_interval)])/2
            # print ("reference = %s" %reference)

            ## Generate arrays of 3D points
            X,Y,Z = np.meshgrid(x_interval, y_interval, z_interval, indexing='ij')    # 'ij' yields vertical x-coordinate for image!
            points = np.array([X.flatten(), Y.flatten(), Z.flatten()])

            ## Compute Gaussian values
            vals = compute_gaussian(points, origin, Sigma)

            ## Reshape to grid defined by X, Y and Z
            Vals = vals.reshape(x_interval.size, y_interval.size, z_interval.size)

            ## Compute coverage of gaussian (i.e. approximation of confidence interval)
            Phi = np.sum(Vals)*step**dim

            ## Normalize values to kernel
            kernel = Vals/np.sum(vals)

        else:
            raise ValueError("Error: Dimension must be 2 or 3")

    
    print("%s-kernel = \n%s" %(kernel.shape,kernel))
    print("np.sum(kernel) = %s" %(np.sum(kernel)))
    print("reference = %s" %reference)
    print("confidence interval coverage by chosen kernel (approx) = %s" %Phi)

    if cutoff.size == 1:
        print("possible confidence interval before eliminating values (approx) = %s" %Phi_all)
        print("cutoff 'radius' for ellipse = %s" %cutoff)


    return kernel, reference




## Smooth image based on kernel
def get_smoothed_2D_image_by_hand(image_sitk, kernel, reference):

    nda = sitk.GetArrayFromImage(image_sitk)
    nda_smoothed = np.zeros(nda.shape)

    left = -reference[1]
    right = kernel.shape[1] - reference[1]

    up = -reference[0]
    down = kernel.shape[0] - reference[0]

    # print("(left, right) = (%s, %s)" %(left,right))
    # print("(up, down) = (%s, %s)" %(up,down))


    ## by hand
    for i in range(0, nda.shape[0]):
        for j in range(0, nda.shape[1]):

            tmp = 0

            for k in range(up, down):
                for l in range(left, right):
                    if ( 0<=i+k and i+k<nda.shape[0] ) \
                        and ( 0<=j+l and j+l<nda.shape[1] ):
                        tmp += nda[i+k,j+l]*kernel[reference[0]+k,reference[1]+l]

            nda_smoothed[i,j] = tmp
        
    image_smoothed_sitk = sitk.GetImageFromArray(nda_smoothed)
    image_smoothed_sitk.CopyInformation(image_sitk)

    # print nda_smoothed.min()
    # print nda_smoothed.max()

    return image_smoothed_sitk


def get_smoothed_image_by_scipy(image_sitk, kernel, reference):

    nda = sitk.GetArrayFromImage(image_sitk)

    ## via scipy:
    ## TODO: https://github.com/scipy/scipy/issues/4580 for correct setting of origin
    origin = np.array(kernel.shape)/2 - reference
    print("Update of origin for scipy.ndimage.convole: origin = %s" % origin)
    nda_smoothed_scipy = ndimage.convolve(nda, kernel, mode='nearest', origin=origin) ##'nearest' resembles results of sitk.SmoothingRecursiveGaussianImageFilter best
    # nda_smoothed_scipy = ndimage.convolve(nda, kernel, mode='mirror', origin=origin) ##'nearest' resembles results of sitk.SmoothingRecursiveGaussianImageFilter best

    image_sitk_smoothed = sitk.GetImageFromArray(nda_smoothed_scipy)
    image_sitk_smoothed.CopyInformation(image_sitk)

    return image_sitk_smoothed


def plot_comparison_of_images(image_sitk_smoothed_by_hand, image_sitk_smoothed_via_scipy):
    nda_smoothed = sitk.GetArrayFromImage(image_sitk_smoothed_by_hand)
    nda_smoothed_scipy = sitk.GetArrayFromImage(image_sitk_smoothed_via_scipy)

    diff = nda_smoothed_scipy-nda_smoothed
    abs_diff = abs(diff)
    norm_diff = np.linalg.norm(diff)

    print("abs_diff_min = %s, abs_diff_max = %s, norm_diff = %s" %(abs_diff.min(), abs_diff.max(), norm_diff))


    ## compare results obtained by hand with those via scipy
    fig = plt.figure()
    plt.subplot(131)
    # plt.imshow(fixed, cmap="Greys_r", origin="low")
    plt.imshow(nda_smoothed, cmap="Greys_r")
    plt.xlabel("nda_smoothed")

    plt.subplot(132)
    # plt.imshow(warped, cmap="Greys_r", origin="low")
    plt.imshow(nda_smoothed_scipy, cmap="Greys_r")
    plt.xlabel("nda_smoothed_scipy")

    plt.subplot(133)
    # plt.imshow(warped, cmap="Greys_r", origin="low")
    plt.imshow(abs_diff, cmap="Greys_r")
    plt.title("abs_diff_min = %s, abs_diff_max = %s, norm_diff = %s" %(abs_diff.min(), abs_diff.max(), norm_diff))
    plt.xlabel("abs_diff")
    # plt.colorbar()

    plt.show(block=False)       # does not pause, but needs plt.draw() (or plt.show()) at end 
                                # of file to be visible

    return fig


## Plot of gaussian
#  \param Sigma variance-covariance matrix \in \R^{2 \times 2}
#  \param origin origin of ellipsoid \in \R^{2}
#  \param x_interval array describing the x-interval
#  \param y_interval array describing the y-interval
#  \param contour_plot either contour plot or heat map can be chosen
def plot_gaussian(Sigma, origin, x_interval, y_interval, contour_plot=1):
    x_interval = np.array(x_interval)
    y_interval = np.array(y_interval)

    ## Generate array of 2D points
    X,Y = np.meshgrid(x_interval,y_interval)
    points = np.array([X.flatten(), Y.flatten()])

    ## Evaluate points
    vals = evaluate_function_ellipsoid(points, origin, Sigma)

    ## Define levels and colours for contour plot
    dim = Sigma.shape[0]
    level_050 = chi2.ppf(0.50, dim)
    level_070 = chi2.ppf(0.70, dim)
    level_080 = chi2.ppf(0.80, dim)
    level_090 = chi2.ppf(0.90, dim)
    level_095 = chi2.ppf(0.95, dim)

    levels = np.array([level_050, level_070, level_080, level_090, level_095])
    levels_name = ['50%', '70%', '80%', '90%', '95%']
    # levels_name = np.array([0.5, 0.7, 0.8, 0.90, 0.95])
    colours = ('blue', 'green', 'orange', 'magenta', 'red')

    vals = np.exp(-0.5*vals)/np.sqrt( (2*np.pi)**dim * np.linalg.det(Sigma) )
    levels = np.exp(-0.5*levels)/np.sqrt( (2*np.pi)**dim * np.linalg.det(Sigma) )
        
    ## Combine levels with corresponding labels for contour lines
    fmt = {}
    for l,s in zip(levels, levels_name):
        fmt[l] =  s

    ## Reshape so that values fit meshgrid structure
    Vals = vals.reshape(y_interval.size,-1)

    ## Plot
    fig = plt.figure()
    plt.scatter(origin[0,0], origin[1,0], s=30, c='yellow')

    # CS = plt.contour(X,Y, Vals, levels)
    CS = plt.contour(X,Y, Vals, levels, colors=colours)

    if contour_plot:
        plt.clabel(CS, inline=True, fmt=fmt, fontsize=10)
    
    else:
        plt.clabel(CS, inline=True, fmt=fmt, fontsize=10)        
        plt.imshow(Vals, origin='lower', extent=[x_interval.min(), x_interval.max(), y_interval.min(), y_interval.max()])
        plt.colorbar()
        
        
    ax = fig.gca()
    ax.set_xticks(np.arange(x_interval.min(),x_interval.max()+1,1))
    ax.set_yticks(np.arange(y_interval.min(),y_interval.max()+1,1))

    plt.grid()
    plt.title("Sigma = %s, origin = %s" %( Sigma.flatten(), origin.flatten() ))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show(block=False)       # does not pause, but needs plt.draw() (or plt.show()) at end 
                                # of file to be visible

    return fig



"""
Unit Test Class
"""
class TestUM(unittest.TestCase):

    accuracy = 7

    def compute_single_value(self, point, origin, Sigma):
        return (point-origin).dot(np.linalg.inv(Sigma)).dot(point-origin)


    def setUp(self):
        pass


    def test_01_check_vectorized_computation_of_evaluate_function_ellipsoid(self):
        
        ## Define points
        points = np.zeros((2,3))
        points[:,0] = (1,1)
        points[:,1] = (2,2)
        points[:,2] = (3,3)

        ## Define origin
        origin = np.zeros((2,1))
        origin[0] = 0
        origin[1] = 1

        ## Define variance covariance matrix
        sigma_x2 = 4
        sigma_y2 = 1
        Sigma = np.identity(2)
        Sigma[0,0] = sigma_x2
        Sigma[0,1] = 1
        Sigma[1,1] = sigma_y2
        Sigma[1,0] = Sigma[0,1]

        N = points.shape[1]

        ## Compute values separately
        vals_0 = np.zeros(N)
        for i in range(0, N):
            vals_0[i] = self.compute_single_value(points[:,i], origin[:,0], Sigma)

        ## Compute values in vectorized version
        vals_1 = evaluate_function_ellipsoid(points, origin, Sigma)

        ## Check results      
        self.assertEqual(np.around(
            np.linalg.norm( vals_0 - vals_1) 
            , decimals = self.accuracy), 0 )


    ## Square kernel (3 x 3)
    def test_02_2D_compare_smoothing_results_of_scipy_and_by_hand_square_kernel(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)

        kernel = np.array([\
            [ 0.08435869,  0.13908396,  0.08435869],
            [ 0.10535125,  0.17369484,  0.10535125],
            [ 0.08435869,  0.13908396,  0.08435869]])

        reference = np.array([1,1])

        image_smoothed_sitk = get_smoothed_2D_image_by_hand(image_sitk, kernel, reference)
        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)

        nda_hand = sitk.GetArrayFromImage(image_smoothed_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_hand-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## Rectangular kernel (3 x 5)
    def test_02_2D_compare_smoothing_results_of_scipy_and_by_hand_rectangular_kernel(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)

        kernel = np.array([\
            [ 0.06935881,  0.10347119,  0.06935881,  0.02089047,  0.00282722],\
            [ 0.03444257,  0.11435335,  0.17059515,  0.11435335,  0.03444257],
            [ 0.00282722,  0.02089047,  0.06935881,  0.10347119,  0.06935881]])

        reference = np.array([1,2])

        image_smoothed_sitk = get_smoothed_2D_image_by_hand(image_sitk, kernel, reference)
        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)

        nda_hand = sitk.GetArrayFromImage(image_smoothed_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_hand-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## Rectangular kernel (3 x 5)
    def test_02_2D_compare_smoothing_results_of_scipy_and_by_hand_rectangular_kernel_altered_reference(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)

        kernel = np.array([\
            [ 0.06935881,  0.10347119,  0.06935881,  0.02089047,  0.00282722],\
            [ 0.03444257,  0.11435335,  0.17059515,  0.11435335,  0.03444257],
            [ 0.00282722,  0.02089047,  0.06935881,  0.10347119,  0.06935881]])

        reference = np.array([1,1])

        image_smoothed_sitk = get_smoothed_2D_image_by_hand(image_sitk, kernel, reference)
        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)

        nda_hand = sitk.GetArrayFromImage(image_smoothed_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_hand-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.65
    ##   - Sigma = [1, 0; 0 1.5**2]
    def test_02_2D_compare_smoothing_results_of_scipy_and_by_hand_elliptic_kernel(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)

        kernel = np.array([\
            [ 0.        ,  0.0738165 ,  0.09218565,  0.0738165 ,  0.        ],\
            [ 0.06248431,  0.12170283,  0.15198844,  0.12170283,  0.06248431],\
            [ 0.        ,  0.0738165 ,  0.09218565,  0.0738165 ,  0.        ]])
        reference = np.array([1,2])

        image_smoothed_sitk = get_smoothed_2D_image_by_hand(image_sitk, kernel, reference)
        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)

        nda_hand = sitk.GetArrayFromImage(image_smoothed_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_hand-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.65
    ##   - Sigma = [1, 1; 1 1.5**2]
    def test_02_2D_compare_smoothing_results_of_scipy_and_by_hand_elliptic_kernel_skewed(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)

        kernel = np.array([\
            [ 0.07848865,  0.11709131,  0.07848865,  0.        ,  0.        ],\
            [ 0.        ,  0.12940591,  0.19305094,  0.12940591,  0.        ],\
            [ 0.        ,  0.        ,  0.07848865,  0.11709131,  0.07848865]])
        reference = np.array([1,2])

        image_smoothed_sitk = get_smoothed_2D_image_by_hand(image_sitk, kernel, reference)
        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)

        nda_hand = sitk.GetArrayFromImage(image_smoothed_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_hand-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.65
    ##   - Sigma = [1, 1; 1 1.5**2]
    def test_02_2D_compare_smoothing_results_of_scipy_and_by_hand_elliptic_kernel_skewed_altered_reference(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)

        kernel = np.array([\
            [ 0.07848865,  0.11709131,  0.07848865,  0.        ,  0.        ],\
            [ 0.        ,  0.12940591,  0.19305094,  0.12940591,  0.        ],\
            [ 0.        ,  0.        ,  0.07848865,  0.11709131,  0.07848865]])
        reference = np.array([1,3])

        image_smoothed_sitk = get_smoothed_2D_image_by_hand(image_sitk, kernel, reference)
        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)

        nda_hand = sitk.GetArrayFromImage(image_smoothed_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_hand-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.95
    ##   - Sigma = [5**2, 0; 0 5**2]
    def test_02_2D_compare_smoothing_results_of_scipy_and_recursive_gaussian_elliptic_kernel(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        sigma = 5

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)

        kernel = np.loadtxt(dir_input+"kernel_2D_Sigma_" + str(sigma**2) + "_0_0_" + str(sigma**2) + ".txt")
        reference = np.array(kernel.shape)/2

        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)
        image_smoothed_recursive_sitk = gaussian.Execute(image_sitk)
        
        nda_recursive = sitk.GetArrayFromImage(image_smoothed_recursive_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_recursive-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.95
    ##   - Sigma = [5**2, 0; 0 5**2]
    ##   - Spacing = (1.5, 3)
    def test_02_2D_compare_smoothing_results_of_scipy_and_recursive_gaussian_elliptic_kernel_nonuniform_spacing(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        sigma = 5
        spacing = (1.5, 3)

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)
        image_sitk.SetSpacing(spacing)

        kernelname = "kernel_2D_Sigma_" + str(sigma**2) + "_0_0_" + str(sigma**2) 
        kernelname += "_spacing_" + str(spacing[0]) + "_" + str(spacing[1])
        kernel = np.loadtxt(dir_input + kernelname + ".txt")
        reference = np.array(kernel.shape)/2

        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)
        image_smoothed_recursive_sitk = gaussian.Execute(image_sitk)
        
        nda_recursive = sitk.GetArrayFromImage(image_smoothed_recursive_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_recursive-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())



    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.95
    ##   - Sigma = [3**2, 0, 0; 0 3**2 0; 0, 0, 3**2]
    ##   - Unit spacing
    def test_03_3D_compare_smoothing_results_of_scipy_and_recursive_gaussian_elliptic_kernel_unit_spacing(self):

        dir_input = "data/"
        filename =  "fetal_brain_a"
        image_type = ".nii.gz"

        sigma = 3
        spacing = (1, 1, 1)

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)
        image_sitk.SetSpacing(spacing)

        kernelname = "kernel_3D_Sigma_9_0_0_0_9_0_0_0_9_spacing_1_1_1"

        kernel = np.load(dir_input + kernelname + ".npy")
        reference = np.array(kernel.shape)/2

        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)
        image_smoothed_recursive_sitk = gaussian.Execute(image_sitk)
        
        nda_recursive = sitk.GetArrayFromImage(image_smoothed_recursive_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_recursive-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.95
    ##   - Sigma = [3**2, 0, 0; 0 3**2 0; 0, 0, 3**2]
    ##   - Uniform spacing (1.7)
    def test_03_3D_compare_smoothing_results_of_scipy_and_recursive_gaussian_elliptic_kernel_uniform_spacing_1(self):

        dir_input = "data/"
        filename =  "fetal_brain_a"
        image_type = ".nii.gz"

        sigma = 3
        spacing = (1.7, 1.7, 1.7)

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)
        image_sitk.SetSpacing(spacing)

        kernelname = "kernel_3D_Sigma_9_0_0_0_9_0_0_0_9_spacing_1.7_1.7_1.7"

        kernel = np.load(dir_input + kernelname + ".npy")
        reference = np.array(kernel.shape)/2

        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)
        image_smoothed_recursive_sitk = gaussian.Execute(image_sitk)
        
        nda_recursive = sitk.GetArrayFromImage(image_smoothed_recursive_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_recursive-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.95
    ##   - Sigma = [3**2, 0, 0; 0 3**2 0; 0, 0, 3**2]
    ##   - Uniform spacing (3)
    def test_03_3D_compare_smoothing_results_of_scipy_and_recursive_gaussian_elliptic_kernel_uniform_spacing_2(self):

        dir_input = "data/"
        filename =  "fetal_brain_a"
        image_type = ".nii.gz"

        sigma = 3
        spacing = (3, 3, 3)

        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)
        image_sitk.SetSpacing(spacing)

        kernelname = "kernel_3D_Sigma_9_0_0_0_9_0_0_0_9_spacing_3_3_3"

        kernel = np.load(dir_input + kernelname + ".npy")
        reference = np.array(kernel.shape)/2

        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)
        image_smoothed_recursive_sitk = gaussian.Execute(image_sitk)
        
        nda_recursive = sitk.GetArrayFromImage(image_smoothed_recursive_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_recursive-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.95
    ##   - Sigma = [3**2, 0, 0; 0 3**2 0; 0, 0, 3**2]
    ##   - original spacing
    """
    PROBLEM!!!
    """
    def test_03_3D_compare_smoothing_results_of_scipy_and_recursive_gaussian_elliptic_kernel(self):

        dir_input = "data/"
        filename =  "fetal_brain_a"
        image_type = ".nii.gz"

        sigma = 3
        ## Read image
        #  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
        image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)

        kernelname = "kernel_3D_Sigma_9_0_0_0_9_0_0_0_9"

        kernel = np.load(dir_input + kernelname + ".npy")
        reference = np.array(kernel.shape)/2

        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)
        image_smoothed_recursive_sitk = gaussian.Execute(image_sitk)
        
        nda_recursive = sitk.GetArrayFromImage(image_smoothed_recursive_sitk)
        nda_scipy = sitk.GetArrayFromImage(image_smoothed_sitk_scipy)

        diff = nda_recursive-nda_scipy
        abs_diff = abs(diff)
        norm_diff = np.linalg.norm(diff)


        ## Check results      
        try:
            self.assertEqual(np.around(
                norm_diff
                , decimals = self.accuracy), 0 )

        except Exception as e:
            print("FAIL: " + self.id() + " failed given norm of difference = %.2e > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


"""
Main
"""
## Specify data
dir_input = "data/"
dir_output = "results/"

# filename =  "placenta_s"
# filename =  "kidney_s"
# filename =  "fetal_brain_c"
# filename =  "fetal_brain_s"

filename = "BrainWeb_2D"
image_type = ".png"

filename =  "fetal_brain_a"
image_type = ".nii.gz"

## Read image
#  (Float32 is default in sitk.SmoothingRecursiveGaussianImageFilter. Hence, use this data type for comparisons)
image_sitk = sitk.ReadImage(dir_input + filename + image_type, sitk.sitkFloat32)
# image_sitk = image_sitk[:,:,8]

dim = image_sitk.GetDimension()

Sigma = np.identity(dim)
origin = np.zeros((dim,1))


if dim == 2:
    sigma_x = 5
    sigma_y = 5

    Sigma[0,0] = sigma_x**2
    # Sigma[0,1] = 1
    Sigma[1,1] = sigma_y**2
    Sigma[1,0] = Sigma[0,1]

    # origin[0] = 0.1
    # origin[1] = 0.4

    kernel_size = (3,5)

    # t = 2
    # image_sitk.SetSpacing((t,t))  
    image_sitk.SetSpacing((1.5,3))  

elif dim == 3:
    sigma_x = 3
    sigma_y = 3
    sigma_z = 3

    Sigma[0,0] = sigma_x**2
    # Sigma[0,1] = 2
    # Sigma[0,2] = 1
    Sigma[1,1] = sigma_y**2
    # Sigma[1,2] = 1
    Sigma[2,2] = sigma_z**2
    Sigma[1,0] = Sigma[0,1]
    Sigma[2,0] = Sigma[0,2]
    Sigma[2,1] = Sigma[1,2]

    # origin[0] = 1
    # origin[1] = 4
    # origin[2] = -1

    kernel_size = (3,5,3)

    ## Unit spacing for better comparison
    t = 3
    image_sitk.SetSpacing((t,t,t))  



try:
    if is_SPD(Sigma):
        print("Sigma = \n%s" %Sigma)

        spacing = image_sitk.GetSpacing()

        ## Generate arrays of 2D points
        if dim == 2:
            step = 1
            x_interval = np.arange(-1,2,step)
            y_interval = np.arange(-1,2,step)
            X,Y = np.meshgrid(x_interval, y_interval, indexing='ij')    # 'ij' yields vertical x-coordinate for image!

            points = np.array([X.flatten(), Y.flatten()])

            # print("x_interval = %s" %x_interval)
            # print("y_interval = %s" %y_interval)
            # print("X = \n%s" %X)
            # print("Y = \n%s" %Y)
            # print("points = \n%s" %points)
            # print("origin = \n%s" %origin)


        ## Generate arrays of 3D points
        elif dim == 3:
            step = 1
            x_interval = np.arange(-1,1,step)
            y_interval = np.arange(-2,0,step)
            z_interval = np.arange(-3,-1,step)
            X,Y,Z = np.meshgrid(x_interval, y_interval, z_interval, indexing='ij')
            points = np.array([X.flatten(), Y.flatten(), Z.flatten()])

            # print("x_interval = %s" %x_interval)
            # print("y_interval = %s" %y_interval)
            # print("z_interval = %s" %z_interval)
            # print("X = \n%s" %X)
            # print("Y = \n%s" %Y)
            # print("Z = \n%s" %Z)
            # print("points = \n%s" %points)
            # print("origin = \n%s" %origin)


        ## Get contour line level:
        alpha = 0.95
        cutoff_level = chi2.ppf(alpha, dim)
        print("cutoff_level = %s" %cutoff_level)

        ## Check whether points are within ellipsoid
        # print is_in_ellipsoid(points[:,0], origin, Sigma, cutoff_level)

        kernel, reference = get_smoothing_kernel(image_sitk, Sigma, origin, cutoff=cutoff_level)
        # kernel, reference = get_smoothing_kernel(image_sitk, Sigma, origin, cutoff=kernel_size)

        save_kernel = 0
        ## Used for later read-in for unit tests
        if save_kernel:
            ## write variance covariance matrix
            name = "kernel_" + str(dim) + "D_Sigma_" + str(Sigma.flatten()[0].astype("uint8"))
            for i in range(1, dim**2):
                name = name + "_" + str(Sigma.flatten()[i].astype("uint8"))

            ## write spacing information
            name = name + "_spacing_" + str(spacing[0])
            for i in range(1, dim):
                name = name + "_" + str(spacing[1])

            ## write as txt-file
            if dim==2:
                np.savetxt(dir_output +  name + ".txt", kernel)

            ## np.savetxt does not work for more than 2 dimensions. Hence np.save => *.npy file
            elif dim==3:
                np.save(dir_output +  name , kernel)


        # image_smoothed_sitk = get_smoothed_2D_image_by_hand(image_sitk, kernel, reference)
        image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)

        # plot_comparison_of_images(image_smoothed_sitk, image_smoothed_sitk_scipy)


        # sitkh.show_sitk_image(image_smoothed_sitk)

        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        # gaussian.SetSigma(sigma_x/spacing[0]) #Sigma is measured in image spacing units!?
        gaussian.SetSigma(sigma_x) #Sigma is measured in image spacing units!?
        image_smoothed_recursive_sitk = gaussian.Execute(image_sitk)
        
        # plot_comparison_of_images(image_smoothed_recursive_sitk, image_smoothed_sitk_scipy)

        # sitkh.show_sitk_image(image_sitk=image_smoothed_sitk, overlay_sitk=image_smoothed_recursive_sitk)
        # sitkh.show_sitk_image(image_sitk=image_smoothed_sitk)
        # sitkh.show_sitk_image(image_sitk=image_smoothed_sitk_scipy, overlay_sitk=image_smoothed_recursive_sitk)

        diff_sitk = image_smoothed_sitk_scipy-image_smoothed_recursive_sitk
        sitkh.show_sitk_image(diff_sitk)

        abs_diff = np.abs(sitk.GetArrayFromImage(diff_sitk))
        print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
        print("     Mean absolute difference per voxel: %s" %abs_diff.mean())
        print("     Minimum absolute difference per voxel: %s" %abs_diff.min())



        ## Scale variance-covariance matrix by given factor
        Sigma_scale = get_scaled_variance_covariance_matrix(Sigma, 1/np.array(image_sitk.GetSpacing()))

        ## Plot 
        if dim == 2:

            M_x = 3*sigma_x
            M_y = 3*sigma_y
            step = 0.1
            x_interval = np.arange(-M_x,M_x+step,step)
            y_interval = np.arange(-M_y,M_y+step,step)

            # plot_gaussian(Sigma_scale, origin, x_interval, y_interval, contour_plot=0)

        print("alpha = %s" %alpha)

        plt.draw()        

    else:
        raise ValueError("Error: Sigma is not SPD")


except ValueError as err:
    print("Sigma = \n%s" %Sigma)
    print("eigenvalues(Sigma) = \n%s" %np.linalg.eigvals(Sigma))
    print(err.args[0])





"""
Unit tests:
"""
print("\nUnit tests:\n--------------")
unittest.main()
