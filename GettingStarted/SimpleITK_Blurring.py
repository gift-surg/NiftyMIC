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
        if origin.size is 2 or orign.size is 3:
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
#  \param origin origin of ellipsoid \in \R^{dim}
#  \param Sigma variance-covariance matrix \in \R^{dim \times dim}
#  \return values of (multivariate) Gaussian distribution
def compute_gaussian(Sigma, mu, points):

    points = np.array(points)
    origin = np.array(origin)

    dim = Sigma.shape[0]

    try:
        ## Guarantee that origin is column vector, i.e. in \R^{(2 or 3)}
        if origin.size is 2 or orign.size is 3:
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
    U,s,V = np.linalg.svd(Sigma)

    ## Scale variances
    # print("Variances before scaling with factor(s)=%s: %s" %(scales,s))
    s = scales*s;
    # print("Variances after scaling with factor(s)=%s: %s" %(scales,s))

    ## Computed scaled variance covariance matrix
    Sigma = U.dot(np.diag(s)).dot(np.transpose(V))

    return Sigma


## Plot of gaussian
#  \param Sigma variance-covariance matrix \in \R^{2 \times 2}
#  \param origin origin of ellipsoid \in \R^{2}
#  \param x_interval array describing the x-interval
#  \param y_interval array describing the y-interval
#  \param contour_plot either contour plot or heat map can be chosen
#  \param scaled scale to gaussian distribution
def plot_gaussian(Sigma, origin, x_interval, y_interval, contour_plot=1, scaled=1):
    x_interval = np.array(x_interval)
    y_interval = np.array(y_interval)

    ## Generate array of 2D points
    X,Y = np.meshgrid(x_interval,y_interval)
    points = np.array([X.flatten(), Y.flatten()])

    ## Evaluate points
    vals = evaluate_function_ellipsoid(points, origin, Sigma)

    if scaled:
        vals = np.exp(-0.5*vals)/( (2*np.pi)**1 * np.sqrt(np.linalg.det(Sigma)) )

    ## Reshape so that values fit meshgrid structure
    Vals = vals.reshape(x_interval.size,-1)

    ## Define levels and colours for contour plot
    levels = np.array([0.1, 0.5, 1, 2, 5])
    colours = ('orange', 'orange', 'red', 'blue', 'blue')

    if scaled:
        levels = np.exp(-0.5*levels)/( (2*np.pi)**1 * np.sqrt(np.linalg.det(Sigma)) )

    ## Plot
    fig = plt.figure()
    plt.scatter(origin[0,0], origin[1,0], s=30, c='yellow')

    if contour_plot:
        CS = plt.contour(X,Y, Vals, levels, colors=colours)
        plt.clabel(CS, inline=1, fontsize=10)
    
    else:
        CS = plt.contour(X,Y, Vals, levels, colors=colours)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.imshow(Vals, origin='lower', extent=[x_interval.min(), x_interval.max(), y_interval.min(), y_interval.max()])
        plt.colorbar()
        
        
    ax = fig.gca()
    ax.set_xticks(np.arange(x_interval.min(),x_interval.max()+1,1))
    ax.set_yticks(np.arange(y_interval.min(),y_interval.max()+1,1))

    plt.grid()
    plt.title("Sigma = %s, origin = %s" %( Sigma.flatten(), origin.flatten() ))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    return None


## get smoothing kernel
#  \param image_sitk
#  \param Sigma
#  \param origin
#  \param cutoff either
#  \return kernel 
#  \return reference
def get_smoothing_kernel(image_sitk, Sigma, origin, cutoff):

    try:
        cutoff = np.array(cutoff)

        spacing = np.array(image_sitk.GetSpacing())

        ## Scale to image space with cutoff-sigma
        scaling = 1/spacing

        Sigma_image = get_scaled_variance_covariance_matrix(Sigma, scaling)
        
        ## Perform SVD
        U,s_image,V = np.linalg.svd(Sigma_image)

        # print s_image
        # U,s,V = np.linalg.svd(Sigma)
        # print s

        ## cutoff = cutoff_level, i.e. a scalar to determine cutoff level of ellipse
        if cutoff.size == 1:
            ## Maximumg length of vector in image space
            ## TODO: more exact
            l_max_image = np.sqrt(np.linalg.norm(s_image)*cutoff)
            print("l_max_image = %s" %(l_max_image))
            l_max_image = np.ceil(l_max_image)

            ## Generate intervals for x and y based on l_max_image
            step = 1
            x_interval = np.arange(-l_max_image, l_max_image+step,step)
            y_interval = np.arange(-l_max_image, l_max_image+step,step)

        ## cutoff represents kernel size
        else:
            # if cutoff[0]%2==0 or cutoff[1]%2==0:
            #     raise ValueError("Error: kernel size must consist of odd numbers in both dimensions")

            ## create intervalls
            step = 1
            x_interval = np.arange(0,cutoff[0],step)
            y_interval = np.arange(0,cutoff[1],step)

            ## symmetry around zero
            x_interval = x_interval-x_interval.mean()
            y_interval = y_interval-y_interval.mean()


        ## Store reference/center/midpoint of kernel
        # print x_interval
        # origin = reference = np.array(kernel.shape)/2

        reference = np.array([len(x_interval), len(y_interval)])/2
        # reference = np.array([2,2])
        # print ("reference = %s" %reference)

        ## Generate arrays of 2D points
        Y,X = np.meshgrid(y_interval,x_interval)    # do it this order so that x-coordinate is vertical for image!
        points = np.array([X.flatten(), Y.flatten()])

        ## cutoff = cutoff_level, i.e. a scalar to determine cutoff level of ellipse
        if cutoff.size == 1:

            ## Determine points in ellipsoid
            bool, vals = is_in_ellipsoid(points, origin, Sigma_image, cutoff)

            ## Compute Gaussian values
            dim = Sigma_image.shape[0]
            vals = np.exp(-0.5*vals)/np.sqrt((2*np.pi)**dim * np.linalg.det(Sigma_image)) 

            ## Reshape to grid defined by X and Y
            Bool = bool.reshape(x_interval.size,-1) 
            Vals = vals.reshape(x_interval.size,-1) 

            # print("X = \n%s" %(X))
            # print("Y = \n%s" %(Y))
            # print("Bool = \n%s" %(Bool))
            # print("Vals = \n%s" %(Vals))

            ## Normalize values of kernel
            Phi_all = np.sum(Vals)*step**2

            ## Set values out of desired ellipse to zero
            Vals[Bool==0] = 0

            ## Compute coverage of gaussian (i.e. approximation of confidence interval)
            Phi = np.sum(Vals[Bool])*step**2

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
            X = np.delete(X, delete_ind_rows, 0)
            Y = np.delete(Y, delete_ind_rows, 0)
            kernel = np.delete(kernel, delete_ind_rows, 0)
            Bool = np.delete(Bool, delete_ind_rows, 0)

            X = np.delete(X, delete_ind_cols, 1)
            Y = np.delete(Y, delete_ind_cols, 1)
            kernel = np.delete(kernel, delete_ind_cols, 1)
            Bool = np.delete(Bool, delete_ind_cols, 1)

            # print("cutoff = %s" %cutoff)
            # print("Bool = \n%s" %(Bool))
            # print("X = \n%s" %(X))
            # print("Y = \n%s" %(Y))


        ## cutoff represents kernel size
        else:
            ## Evaluate equation for ellipsoid
            vals = evaluate_function_ellipsoid(points, origin, Sigma)

            ## Compute Gaussian values
            dim = Sigma_image.shape[0]
            vals = np.exp(-0.5*vals)/np.sqrt((2*np.pi)**dim * np.linalg.det(Sigma_image)) 

            ## Reshape to grid defined by X and Y
            Vals = vals.reshape(x_interval.size,-1)

            ## Compute coverage of gaussian (i.e. approximation of confidence interval)
            Phi = np.sum(Vals)*step**2

            ## Normalize values to kernel
            kernel = Vals/np.sum(vals)


        
        print("(%sx%s)-kernel = \n%s" %(kernel.shape[0], kernel.shape[1],kernel))
        print("np.sum(kernel) = %s" %(np.sum(kernel)))
        print("reference = %s" %reference)
        print("confidence interval coverage by chosen kernel = %s" %Phi)

        if cutoff.size == 1:
            print("cutoff 'radius' for ellipse = %s" %cutoff)
            print("possible confidence interval before eliminating values = %s" %Phi_all)


        return kernel, reference

    except ValueError as err:
        print(err.args[0])
        return None



## Smooth image based on kernel
def get_smoothed_image_by_hand(image_sitk, kernel, reference):

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
    nda_smoothed_scipy = ndimage.convolve(nda, kernel, mode='constant', origin=origin)

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

    plt.show()



def simple_gaussian_2D(Sigma, mu, x, y):
    sigma_x2 = Sigma[0,0]
    sigma_y2 = Sigma[1,1]

    val = np.exp(-0.5*( (x-mu[0])**2/sigma_x2 + (y-mu[1])**2/sigma_y2 ))

    return val


"""
Unit Test Class
"""
class TestUM(unittest.TestCase):

    accuracy = 8

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
    def test_02_compare_smoothing_results_of_scipy_and_by_hand_square_kernel(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        image_sitk = sitk.ReadImage(dir_input + filename + image_type)  

        kernel = np.array([\
            [ 0.08435869,  0.13908396,  0.08435869],
            [ 0.10535125,  0.17369484,  0.10535125],
            [ 0.08435869,  0.13908396,  0.08435869]])

        reference = np.array([1,1])

        image_smoothed_sitk = get_smoothed_image_by_hand(image_sitk, kernel, reference)
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
            print("FAIL: " + self.id() + " failed given norm of difference = %.2f > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## Rectangular kernel (3 x 5)
    def test_02_compare_smoothing_results_of_scipy_and_by_hand_rectangular_kernel(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        image_sitk = sitk.ReadImage(dir_input + filename + image_type)  

        kernel = np.array([\
            [ 0.06935881,  0.10347119,  0.06935881,  0.02089047,  0.00282722],\
            [ 0.03444257,  0.11435335,  0.17059515,  0.11435335,  0.03444257],
            [ 0.00282722,  0.02089047,  0.06935881,  0.10347119,  0.06935881]])

        reference = np.array([1,2])

        image_smoothed_sitk = get_smoothed_image_by_hand(image_sitk, kernel, reference)
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
            print("FAIL: " + self.id() + " failed given norm of difference = %.2f > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## Rectangular kernel (3 x 5)
    def test_02_compare_smoothing_results_of_scipy_and_by_hand_rectangular_kernel_altered_reference(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        image_sitk = sitk.ReadImage(dir_input + filename + image_type)  

        kernel = np.array([\
            [ 0.06935881,  0.10347119,  0.06935881,  0.02089047,  0.00282722],\
            [ 0.03444257,  0.11435335,  0.17059515,  0.11435335,  0.03444257],
            [ 0.00282722,  0.02089047,  0.06935881,  0.10347119,  0.06935881]])

        reference = np.array([1,1])

        image_smoothed_sitk = get_smoothed_image_by_hand(image_sitk, kernel, reference)
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
            print("FAIL: " + self.id() + " failed given norm of difference = %.2f > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.65
    ##   - Sigma = [1, 0; 0 1.5**2]
    def test_02_compare_smoothing_results_of_scipy_and_by_hand_elliptic_kernel(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        image_sitk = sitk.ReadImage(dir_input + filename + image_type)  

        kernel = np.array([\
            [ 0.        ,  0.0738165 ,  0.09218565,  0.0738165 ,  0.        ],\
            [ 0.06248431,  0.12170283,  0.15198844,  0.12170283,  0.06248431],\
            [ 0.        ,  0.0738165 ,  0.09218565,  0.0738165 ,  0.        ]])
        reference = np.array([1,2])

        image_smoothed_sitk = get_smoothed_image_by_hand(image_sitk, kernel, reference)
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
            print("FAIL: " + self.id() + " failed given norm of difference = %.2f > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.65
    ##   - Sigma = [1, 1; 1 1.5**2]
    def test_02_compare_smoothing_results_of_scipy_and_by_hand_elliptic_kernel_skewed(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        image_sitk = sitk.ReadImage(dir_input + filename + image_type)  

        kernel = np.array([\
            [ 0.07848865,  0.11709131,  0.07848865,  0.        ,  0.        ],\
            [ 0.        ,  0.12940591,  0.19305094,  0.12940591,  0.        ],\
            [ 0.        ,  0.        ,  0.07848865,  0.11709131,  0.07848865]])
        reference = np.array([1,2])

        image_smoothed_sitk = get_smoothed_image_by_hand(image_sitk, kernel, reference)
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
            print("FAIL: " + self.id() + " failed given norm of difference = %.2f > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


    ## kernel based on 
    ##   - elliptic cutoff line with confidence level of alpha=0.65
    ##   - Sigma = [1, 1; 1 1.5**2]
    def test_02_compare_smoothing_results_of_scipy_and_by_hand_elliptic_kernel_skewed_altered_reference(self):

        dir_input = "data/"
        filename = "BrainWeb_2D"
        image_type = ".png"

        ## Read image
        image_sitk = sitk.ReadImage(dir_input + filename + image_type)  

        kernel = np.array([\
            [ 0.07848865,  0.11709131,  0.07848865,  0.        ,  0.        ],\
            [ 0.        ,  0.12940591,  0.19305094,  0.12940591,  0.        ],\
            [ 0.        ,  0.        ,  0.07848865,  0.11709131,  0.07848865]])
        reference = np.array([1,3])

        image_smoothed_sitk = get_smoothed_image_by_hand(image_sitk, kernel, reference)
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
            print("FAIL: " + self.id() + " failed given norm of difference = %.2f > 1e-%s" %(norm_diff,self.accuracy))
            print("     Check statistics of difference: (Maximum absolute difference per voxel might be acceptable)")
            print("     Maximum absolute difference per voxel: %s" %abs_diff.max())
            print("     Minimum absolute difference per voxel: %s" %abs_diff.min())


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
image_sitk = sitk.ReadImage(dir_input + filename + image_type)    

# image_sitk.SetSpacing((1.4, 1.2))

## Set variance matrix
sigma_x = 1
sigma_y = 1.5
kernel_size = (3,5)


# cutoff = kernel_size

dim = image_sitk.GetDimension()
Sigma = np.identity(dim)

Sigma[0,0] = sigma_x**2
Sigma[0,1] = 1
Sigma[1,1] = sigma_y**2
Sigma[1,0] = Sigma[0,1]

# point = np.array([1,1]).reshape(dim,-1)
origin = np.zeros((2,1))
# origin[0] = 0.1
# origin[1] = 0.4


# point = np.array([[0,1],[1,0]])

try:
    if is_SPD(Sigma):
        ## Generate arrays of 2D points
        step = 1
        x_interval = np.arange(-1,2,step)
        y_interval = np.arange(-1,2,step)
        X,Y = np.meshgrid(x_interval,y_interval)
        points = np.array([X.flatten(), Y.flatten()])
        # point = np.zeros((2,3))
        # point[:,0] = (1,1)
        # point[:,1] = (2,2)
        # point[:,2] = (3,3)

        # print points
        # print X
        # print Y

        ## Evaluate equation for ellipsoid
        # vals = np.exp(-0.5*evaluate_function_ellipsoid(points, origin, Sigma))
        # Vals = vals.reshape(x_interval.size,-1)

        # print Vals/np.sum(vals)
        # print vals

        res = np.zeros(points.shape[1])
        for i in range(0, points.shape[1]):
            res[i] = simple_gaussian_2D(Sigma, origin, points[0,i], points[1,i])

        # print res

        ## Check whether points are within ellipsoid
        # print is_in_ellipsoid(point, origin, Sigma, cutoff_level)

        ## Get contour line level:
        alpha = 0.65
        dim = Sigma.shape[0]
        cutoff_level = chi2.ppf(alpha, dim)

        # kernel, reference = get_smoothing_kernel(image_sitk, Sigma, origin, cutoff=cutoff_level)
        # kernel, reference = get_smoothing_kernel(image_sitk, Sigma, origin, cutoff=kernel_size)

        reference = np.array([1,1])

        # image_smoothed_sitk = get_smoothed_image_by_hand(image_sitk, kernel, reference)
        # image_smoothed_sitk_scipy = get_smoothed_image_by_scipy(image_sitk, kernel, reference)

        # plot_comparison_of_images(image_smoothed_sitk, image_smoothed_sitk_scipy)



        # sitkh.show_sitk_image(image_smoothed_sitk)

        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma_x)
        image_smoothed_recursive_sitk = gaussian.Execute(image_sitk)

        # sitkh.show_sitk_image(image_sitk=image_smoothed_sitk, overlay_sitk=image_smoothed_recursive_sitk)
        # sitkh.show_sitk_image(image_sitk=image_smoothed_sitk)


        ## Scale variance-covariance matrix by given factor
        Sigma_scale = get_scaled_variance_covariance_matrix(Sigma, 1/np.array(image_sitk.GetSpacing()))

        ## Plot 
        step = 0.1
        M = 3
        x_interval = np.arange(-M,M+step,step)
        y_interval = np.arange(-M,M+step,step)

        # plot_gaussian(Sigma_scale, origin, x_interval, y_interval, contour_plot=1, scaled=0)

        print("alpha = %s" %alpha)
        

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
