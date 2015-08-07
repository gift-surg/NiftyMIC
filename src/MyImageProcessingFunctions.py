## \file MyImageProcessingFunctions.py
#  \brief Image processing functions
# 
#  \author Michael Ebner
#  \date August 2015


## Import libraries 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import nibabel as nib           # nifti files

## Import other py-files within src-folder
from SliceStack import *
from FileAndImageHelpers import *
from SimilarityMeasures import *


#  \param[in] -degree_x rotation around x-axes in degree
#  \param[in] -degree_y rotation around y-axes in degree
#  \param[in] -degree_z rotation around z-axes in degree
#  \param[in] -translation translation vector
def generate_rigid_transformation_matrix_2d(degree=0, translation=np.array([0,0])):

    ## Generate transformation matrix in homogenous coordinates
    transformation = np.identity(3)

    ## Rotation in degree in-plane
    rotation_angle = degree*np.pi/180

    ## Rotation matrix around z-axes:
    rotation = np.identity(2)
    rotation[0,0] = np.cos(rotation_angle)
    rotation[0,1] = -np.sin(rotation_angle)
    rotation[1,0] = np.sin(rotation_angle)
    rotation[1,1] = np.cos(rotation_angle)

    ## insert rotation and translation into affine transformation matrix
    transformation[0:2,0:2] = rotation
    transformation[0:2,2] = translation

    return transformation


#  \brief Change of basis such that the transformation is w.r.t the center of the image
#         (necessary before applying rotation)
#  \param[in] -array 2D-array
#  \param[in] -transformation (3x3)-transformation matrix 
def get_origin_corrected_transformation_2d(array, transformation):

    shape = array.shape

    ## Change of origin before applying rigid transformation:
    t_x = shape[0]/2
    t_y = shape[1]/2

    ## Generate matrices to perform change of basis
    T1 = np.array([
        [1, 0, t_x],
        [0, 1, t_y],
        [0, 0, 1]])

    T2 = np.array([
        [1, 0, -t_x],
        [0, 1, -t_y],
        [0, 0, 1]])

    ## Transformation matrix with respect to image basis
    return T1.dot(transformation).dot(T2)


#  \param[in] -degree_x rotation around x-axes in degree
#  \param[in] -degree_y rotation around y-axes in degree
#  \param[in] -degree_z rotation around z-axes in degree
#  \param[in] -translation translation vector
def generate_rigid_transformation_matrix_3d(degree_x=0, degree_y=0, degree_z=0, translation=np.array([0,0,0])):

    ## Generate transformation matrix in homogenous coordinates
    transformation = np.identity(4)

    ## For in-plane rotation: only rotation angle around z-axes is relevant
    rotation_angle_x = degree_x*np.pi/180
    rotation_angle_y = degree_y*np.pi/180
    rotation_angle_z = degree_z*np.pi/180

    ## Rotation matrix around x-axes:
    rotation_x = np.identity(3)
    rotation_x[1,1] = np.cos(rotation_angle_x)
    rotation_x[1,2] = np.sin(rotation_angle_x)
    rotation_x[2,1] = -np.sin(rotation_angle_x)
    rotation_x[2,2] = np.cos(rotation_angle_x)

    ## Rotation matrix around y-axes:
    rotation_y = np.identity(3)
    rotation_y[0,0] = np.cos(rotation_angle_y)
    rotation_y[0,2] = -np.sin(rotation_angle_y)
    rotation_y[2,0] = np.sin(rotation_angle_y)
    rotation_y[2,2] = np.cos(rotation_angle_y)

    ## Rotation matrix around z-axes:
    rotation_z = np.identity(3)
    rotation_z[0,0] = np.cos(rotation_angle_z)
    rotation_z[0,1] = np.sin(rotation_angle_z)
    rotation_z[1,0] = -np.sin(rotation_angle_z)
    rotation_z[1,1] = np.cos(rotation_angle_z)

    ## insert rotation and translation into affine transformation matrix
    transformation[0:3,0:3] = rotation_x.dot(rotation_y).dot(rotation_z)
    transformation[0:3,3] = translation

    return transformation


#  \brief Change of basis such that the transformation is w.r.t the center of the image
#         (necessary before applying rotation)
#  \param[in] -array 3D-array
#  \param[in] -transformation (4x4)-transformation matrix 
def get_origin_corrected_transformation_3d(array, transformation):

    # affine = image.get_affine()
    shape = array.shape

    ## Change of origin before applying rigid transformation:
    t_x = shape[0]/2
    t_y = shape[1]/2
    t_z = shape[2]/2
    # t_z = 0

    ## Generate matrices to perform change of basis
    T1 = np.array([
        [1, 0, 0, t_x],
        [0, 1, 0, t_y],
        [0, 0, 1, t_z],
        [0, 0, 0, 1]])

    T2 = np.array([
        [1, 0, 0, -t_x],
        [0, 1, 0, -t_y],
        [0, 0, 1, -t_z],
        [0, 0, 0, 1]])

    ## Transformation matrix with respect to image basis
    return T1.dot(transformation).dot(T2)


    ## Apply rigid transformation to image
    # image.set_affine(np.dot(affine,transformation))

    # print affine
    # print transformation
    # print np.dot(affine,transformation)




#  \brief resampling of 2D images 
#  \param[in] -reference 2D array of reference image intensities
#  \param[in] -floating 2D array of floating image intensities
#  \param[in] -transformation transformation from reference to floating image space
#  \param[in] -order interpolation order for resampling
#  \param[in] -padding intensity value for padding
def resampling(reference, floating, transformation, order=0, padding=0):
    flag_fast_computation = False

    if order==0:
        if flag_fast_computation:
            return None

        ## intuitiv approach (but not fast)
        else:
            # Create an empty image based on the reference image discretisation space
            warped_image_array = np.zeros(reference.shape)
            reference_position = np.array([0,0,1])

            # Iterate over all pixel in the reference image
            # iterate over y-axis:
            for j in range(0,reference.shape[1]):
                reference_position[1] = j

                # iterate over x-axis:
                for i in range(0,reference.shape[0]):
                    reference_position[0] = i

                    # Compute the corresponding position in the floating image space
                    floating_position= transformation.dot(reference_position)
                    
                    # Nearest neighbour interpolation
                    if floating_position[0] >= 0 and \
                        floating_position[1] >= 0 and \
                        floating_position[0] <= floating.shape[0]-1 and \
                        floating_position[1] <= floating.shape[1]-1:

                        floating_position[0] = np.round(floating_position[0])
                        floating_position[1] = np.round(floating_position[1])
                        warped_image_array[i][j] = floating[floating_position[0]][floating_position[1]]
                    else:
                        warped_image_array[i][j] = padding

            return warped_image_array

def optimisation(param,):




## Running example of some test code
def main():
    dir_out = "../results/"
    
    image =  SliceStack(dir_out+"input_data/","1")
    image_array = image.get_data()[:,:,30]

    transformation = generate_rigid_transformation_matrix_2d(degree=90)
    transformation = get_origin_corrected_transformation_2d(image_array, transformation)

    # transformation = generate_rigid_transformation_matrix_3d(degree_z=90)
    # transformation = get_origin_corrected_transformation_3d(image_array, transformation)

    start_image_array = resampling(image_array, image_array, transformation)

    # np.savetxt("../results/input_data/test.txt",transformation)

    rotation_values = np.arange(0,361,45)
    print rotation_values

    SSD = np.zeros(len(rotation_values))
    NMI = np.zeros(len(rotation_values))
    joint_entropy = np.zeros(len(rotation_values))

    # for i in range(0,len(rotation_values)):
    #     deg = rotation_values[i]
    #     transformation = generate_rigid_transformation_matrix_2d(degree=deg)
    #     transformation = get_origin_corrected_transformation_2d(start_image_array, transformation)

    #     warped_image_array = resampling(image_array, start_image_array, transformation)

    #     SSD[i] = ssd(image_array, warped_image_array)
    #     NMI[i] = nmi(image_array, warped_image_array)
    #     # joint_entropy[i] = ssd(image_array, warped_image_array)


    # print SSD
    # print NMI

    plot_comparison_of_reference_and_warped_image(image_array, start_image_array)
    plt.show()

    # plt.figure(2)

    # plt.subplot(121)
    # plt.plot(rotation_values,SSD)
    # plt.title('SSD')

    # plt.subplot(122)
    # plt.plot(rotation_values,NMI)
    # plt.title('NMI')
    

    # print transformation

    # Display the results
    # plt.subplot(2, 1, 1)
    # plt.plot(image_array)

    # plt.subplot(2, 1, 2)
    # plt.plot(warped_image_array)
    # plt.legend(['SSD', 'NMI'], loc='upper left')
    plt.show()

    # display_image(image_array)
    # display_image(warped_image_array)


    


    # plot_joint_histogram(image_array,image_array)
    # plt.show()

if __name__ == "__main__":
    main()