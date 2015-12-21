import SimpleITK as sitk
import numpy as np
import unittest
import matplotlib.pyplot as plt

import sys
sys.path.append("../src")

import SimpleITKHelper as sitkh


"""
Functions
"""
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

        ## Check whether everything is well-defined
        if (points.shape[0] is not 2 and points.shape[0] is not 3) \
            or (points.shape[0] is not origin.size) \
            or (Sigma.shape[0] is not Sigma.shape[1]) \
            or (points.shape[0] is not Sigma.shape[0]):
            raise ValueError("Error: Parameters must be of dimension 2 or 3")

        elif ( not np.all(np.linalg.eigvals(Sigma)>0) )\
            or not ( (Sigma.transpose() == Sigma).all() ) \
            or not ( np.linalg.matrix_rank(Sigma) == Sigma.shape[0] ):
            raise ValueError("Error: Sigma is not SPD")


        else:
            ## Compute value according to equation of ellipsoid
            value = np.sum((points-origin)*np.linalg.inv(Sigma).dot(points-origin), 0)
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
    return values <= cutoff_level**2+eps, values


## Scale axis of ellipsoid defined by the variance covariance matrix
#  \param Sigma variance covariance matrix
#  \param scale factor to multiply with main axis lenghts
#  \return scaled variance covariance matrix
def get_scaled_variance_covariance_matrix(Sigma, scales):

    ## Perform SVD
    U,s,V = np.linalg.svd(Sigma)

    ## Scale variances
    print("Variances before scaling with factor(s)=%s: %s" %(scales,s))
    s = scales*s;
    print("Variances after scaling with factor(s)=%s: %s" %(scales,s))

    ## Computed scaled variance covariance matrix
    Sigma = U.dot(np.diag(s)).dot(np.transpose(V))

    return Sigma


## Plot of gaussian
#  \param Sigma variance-covariance matrix \in \R^{2 \times 2}
#  \param origin origin of ellipsoid \in \R^{2}
#  \param x array describing the x-interval
#  \param y array describing the y-interval
#  \param contour_plot either contour plot or heat map can be chosen
#  \param scaled scale to gaussian distribution
def plot_gaussian(Sigma, origin, x, y, contour_plot=1, scaled=1):
    x = np.array(x)
    y = np.array(y)

    ## Generate array of 2D points
    X,Y = np.meshgrid(x,y)
    points = np.array([X.flatten(), Y.flatten()])

    ## Evaluate points
    vals = evaluate_function_ellipsoid(points, origin, Sigma)

    if scaled:
        vals = np.exp(-0.5*vals)/( (2*np.pi)**1 * np.sqrt(np.linalg.det(Sigma)) )

    ## Reshape so that values fit meshgrid structure
    Vals = vals.reshape(y.size,-1)

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
        plt.imshow(Vals, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar()
        
        
    ax = fig.gca()
    ax.set_xticks(np.arange(x.min(),x.max()+1,1))
    ax.set_yticks(np.arange(y.min(),y.max()+1,1))

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
#  \param cutoff_level
#  \return kernel 
#  \return reference
def get_smoothing_kernel(image_sitk, Sigma, origin, cutoff_level):
    spacing = np.array(image_sitk.GetSpacing())

    ## Scale to image space with cutoff-sigma
    scaling = 1/spacing

    Sigma_image = get_scaled_variance_covariance_matrix(Sigma, scaling)
    
    ## Perform SVD
    U,s_image,V = np.linalg.svd(Sigma_image)

    # print s_image
    # U,s,V = np.linalg.svd(Sigma)
    # print s

    ## Maximumg length of vector in image space
    l_max_image = np.sqrt(np.linalg.norm(s_image))
    print("l_max_image = %s" %(l_max_image))
    l_max_image = np.round(l_max_image)

    ## Generate interval for x and y based on l_max_image
    x = np.arange(-l_max_image, l_max_image+1)
    y = np.arange(-l_max_image, l_max_image+1)

    ## Store reference/center/midpoint of kernel
    reference = np.array([x.size/2, y.size/2]).astype(int)
    print ("reference = %s" %reference)

    ## Generate arrays of 2D points
    X,Y = np.meshgrid(x,y)
    points = np.array([X.flatten(), Y.flatten()])

    ## Determine points in ellipsoid
    bool, vals = is_in_ellipsoid(points, origin, Sigma_image, cutoff_level)
    bool = bool.reshape(y.size,-1)  # reshape to grid
    vals = vals.reshape(y.size,-1)  # reshape to grid

    print bool 
    print vals
    
    ## Compute values proportional to gaussian bell curve
    # vals = np.exp(-0.5*vals) / ( (2*np.pi)**1 * np.sqrt(np.linalg.det(Sigma_image)) )
    vals = np.exp(-0.5*vals)

    ## Normalize values of kernel
    vals[bool==0] = 0
    kernel = vals/np.sum(vals[bool])

    print kernel
    print np.sum(kernel)

    ## Find rows which only contain zeros
    delete_ind_rows = []
    for i in range(0, kernel.shape[0]):
        if np.sum(kernel[i,:]) == 0:
            delete_ind_rows.append(i)

            ## Update center/reference/mitpoint
            if i<reference[0]:
                reference[0] = reference[0]-1

    ## Find cols which only contain zeros
    delete_ind_cols = []
    for i in range(0, kernel.shape[1]):
        if np.sum(kernel[:,i]) == 0:
            delete_ind_cols.append(i)

            ## Update center/reference/mitpoint
            if i<reference[1]:
                reference[1] = reference[1]-1

    ## Delete rows and columns which only contain zeros
    kernel = np.delete(kernel, delete_ind_rows, 0)
    kernel = np.delete(kernel, delete_ind_cols, 1)
    bool = np.delete(bool, delete_ind_rows, 0)
    bool = np.delete(bool, delete_ind_cols, 1)

    print bool
    print kernel
    print ("reference = %s" %reference)

    return kernel, reference


## Smooth image based on kernel
def get_smoothed_image(image_sitk, kernel, reference):

    nda = sitk.GetArrayFromImage(image_sitk)
    nda_smoothed = np.zeros(nda.shape)

    left = -reference[1]
    right = kernel.shape[1] - reference[1]

    up = -reference[0]
    down = kernel.shape[0] - reference[0]

    print("(left, right) = (%s, %s)" %(left,right))
    print("(up, down) = (%s, %s)" %(up,down))

    for i in range(0, nda.shape[0]):
        for j in range(0, nda.shape[1]):

            tmp = 0

            for k in range(up, down):
                for l in range(left, right):
                    if ( 0<=i+k and i+k<nda.shape[0] ) \
                        and ( 0<=j+l and j+l<nda.shape[1] ):
                        tmp += nda[i+k,j+l]*kernel[k,l]

            nda_smoothed[i,j] = tmp
        
    image_smoothed_sitk = sitk.GetImageFromArray(nda_smoothed)
    image_smoothed_sitk.CopyInformation(image_sitk)

    return image_smoothed_sitk


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
sigma_x = 3
sigma_y = 3
cutoff_level = 1


dim = image_sitk.GetDimension()
Sigma = np.identity(dim)

Sigma[0,0] = sigma_x**2
# Sigma[0,1] = 0.5
Sigma[1,1] = sigma_y**2
Sigma[1,0] = Sigma[0,1]

# point = np.array([1,1]).reshape(dim,-1)
origin = np.zeros((2,1))
# origin[0] = 0.1
# origin[1] = 0.4


# point = np.array([[0,1],[1,0]])
point = np.zeros((2,3))
point[:,0] = (1,1)
point[:,1] = (2,2)
point[:,2] = (3,3)

stepsize = 0.1
x = np.arange(-5,5+stepsize,stepsize)
y = np.arange(-5,5+stepsize,stepsize)

## Evaluate equation for ellipsoid
# print evaluate_function_ellipsoid(point, origin, Sigma)

## Check whether points are within ellipsoid
# print is_in_ellipsoid(point, origin, Sigma, cutoff_level)

## Scale variance-covariance matrix by given factor
Sigma_scale = get_scaled_variance_covariance_matrix(Sigma, cutoff_level)

kernel, reference = get_smoothing_kernel(image_sitk, Sigma, origin, cutoff_level)
image_smoothed_sitk = get_smoothed_image(image_sitk, kernel, reference)


# sitkh.show_sitk_image(image_smoothed_sitk)
# sitkh.show_sitk_image(image_sitk=image_sitk, overlay_sitk=image_smoothed_sitk)

## Plot 
# plot_gaussian(Sigma_scale, origin, x, y)

gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
gaussian.SetSigma(sigma_x)

image_smoothed_recursive_sitk = gaussian.Execute(image_sitk)
sitkh.show_sitk_image(image_sitk=image_smoothed_sitk, overlay_sitk=image_smoothed_recursive_sitk)
# sitkh.show_sitk_image(image_sitk=image_smoothed_sitk)





"""
Unit tests:
"""
print("\nUnit tests:\n--------------")
unittest.main()
