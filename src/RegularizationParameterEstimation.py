## \file RegularizationParameterEstimation.py
#  \brief This class serves to estimate the regularization parameter \f$ \alpha \f$
#  in order to minimize
#  \f[
#        \frac{1}{2} \sum_k \Vert M_k y_k - M_k A_k x \Vert_{\ell^2}^2 + \alpha\,\Psi(x) \rightarrow \min_x
#  \f]
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
import time  
from scipy.optimize import curve_fit
from scipy import ndimage
import matplotlib.pyplot as plt
import datetime
# import re               #regular expression

## Import modules from src-folder
import SimpleITKHelper as sitkh
import PSF as psf
import Stack as st
import InverseProblemSolver as ips


## Pixel type of used 3D ITK image
pixel_type = itk.D

## ITK image type 
image_type = itk.Image[pixel_type, 3]

class RegularizationParameterEstimation:

    ## Constructor
    #  \param[in] stacks list of Stack objects containing all stacks used for the reconstruction
    #  \param[in] HR_volume Stack object containing the current estimate of the HR volume (used as initial value + space definition)
    def __init__(self, stacks, HR_volume):

        ## Initialize variables
        self._stacks = stacks
        self._HR_volume = HR_volume
        self._N_stacks = len(stacks)

        self._alphas = None
        self._rho = None
        self._residuals = None
        self._Psis = None
        self._regularization_types = None

        self._ADMM_iterations = 5   # Number of performed ADMM iterations

        ## Parameters for output
        # self._filename_prefix = ""
        self._filename_prefix = "RegularizationParameterEstimation_"
        self._dir_results = "RegularizationParameterEstimation/"

        """
        Member functions to compute residual and prior term Psi
        """
        ## Used for PSF modelling
        self._psf = psf.PSF()

        ## Cut-off distance for Gaussian blurring filter
        self._alpha_cut = 3     

        ## Allocate and initialize Oriented Gaussian Interpolate Image Filter
        self._filter_oriented_Gaussian_interpolator = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        self._filter_oriented_Gaussian_interpolator.SetAlpha(self._alpha_cut)
        
        self._filter_oriented_Gaussian = itk.ResampleImageFilter[image_type, image_type].New()
        self._filter_oriented_Gaussian.SetInterpolator(self._filter_oriented_Gaussian_interpolator)
        self._filter_oriented_Gaussian.SetDefaultPixelValue( 0.0 )

        ## Create PyBuffer object for conversion between NumPy arrays and ITK images
        self._itk2np = itk.PyBuffer[image_type]

        ## Extract information ready to use for itk image conversion operations
        self._HR_shape_nda = sitk.GetArrayFromImage( self._HR_volume.sitk ).shape
        self._HR_origin_itk = self._HR_volume.sitk.GetOrigin()
        self._HR_spacing_itk = self._HR_volume.sitk.GetSpacing()
        self._HR_direction_itk = sitkh.get_itk_direction_from_sitk_image( self._HR_volume.sitk )

        ## Define dictionary to choose between computations of prior term \f$ \Psi(x) \f$
        self._compute_prior_term = {
            "TK0"   : self._compute_prior_term_TK0,
            "TK1"   : self._compute_prior_term_TK1,
            "TV-L2" : self._compute_prior_term_TVL2 
        }


    ## Set directory for output results
    #  \param[in] dir_results string
    def set_directory_results(self, dir_results):
        self._dir_results = dir_results


    ## Set prefix for all output results written to directory_results
    #  \param[in] filename_prefix string
    def set_filename_prefix(self, filename_prefix):
        self._filename_prefix = filename_prefix

    ## Define array of regularization parameters which will be used for
    #  the computation
    #  \param[in] alpha_array numpy array of alpha values
    def set_alpha_array(self, alpha_array):
        self._alphas = alpha_array


    ## Get chosen values for alpha
    #  \return numpy array of alpha values
    def get_alpha_array(self):
        return self._alphas


    ## Set rho to be computed in case of TV-L2 regularization
    #  \param[in] rho
    def set_rho(self, rho):
        self._rho = rho




    ## Set SRR approaches which will be run
    #  \param[in] regularization_types list of strings containing only "TK0", "TK1"
    def set_regularization_types(self, regularization_types):
        self._regularization_types = regularization_types


    ## Get chosen types of SRR
    #  \return list of strings
    def get_regularization_types(self):
        return self._regularization_types


    ## Set ADMM iterations to solve TV-L2 reconstruction problem
    #  \[$
    #   \sum_{k=1}^K \frac{1}{2} \Vert y_k - A_k x \Vert_{\ell^2}^2 + \alpha\,\Psi(x) 
    #   + \mu \cdot (\nabla x - v) + \frac{\rho}{2} \Vert \nabla x - v \Vert_{\ell^2}^2
    #  \]$
    #  \param[in] iterations number of ADMM iterations, scalar
    def set_ADMM_iterations(self, iterations):
        self._ADMM_iterations = iterations


    ## Get chosen value of ADMM iterations to solve TV-L2 reconstruction problem
    #  \return number of ADMM iterations, scalar
    def get_ADMM_iterations(self):
        return self._ADMM_iterations


    ## Run all reconstructions for all chosen regularization types and
    #  associated regularization parameters alpha
    def run_reconstructions(self, save_flag=0):

        ## Create output directory in case it is not existing
        os.system("mkdir -p " + self._dir_results)

        N_alphas = len(self._alphas)
        N_regularization_types = len(self._regularization_types)

        self._Psis = np.zeros((N_regularization_types, N_alphas))
        self._residuals = np.zeros((N_regularization_types, N_alphas))

        ## Cut-off distance for Gaussian blurring filter
        SRR_alpha_cut = 3 


        for i_reg_type in range(0, N_regularization_types):

            ## Read type of regularization
            SRR_approach = self._regularization_types[i_reg_type]

            ## Set parameters based on chosen regularization
            if SRR_approach in ["TK0"]:
                SRR_tolerance = 1e-5
                SRR_iter_max = 30
                SRR_DTD_computation_type = "FiniteDifference" #not used
                SRR_rho = None
                SRR_ADMM_iterations = None
                SRR_ADMM_iterations_output_dir = None
                SRR_ADMM_iterations_output_filename_prefix = None

            elif SRR_approach in ["TK1"]:
                SRR_tolerance = 1e-5
                SRR_iter_max = 30
                SRR_DTD_computation_type = "FiniteDifference"
                SRR_rho = None
                SRR_ADMM_iterations = None
                SRR_ADMM_iterations_output_dir = None
                SRR_ADMM_iterations_output_filename_prefix = None

            elif SRR_approach in ["TV-L2"]:
                SRR_tolerance = 1e-3
                SRR_iter_max = 10
                SRR_DTD_computation_type = "FiniteDifference"
                SRR_rho = self._rho       #0.5 yields visually good results
                SRR_ADMM_iterations = self._ADMM_iterations
                SRR_ADMM_iterations_output_dir = self._dir_results + "TV-L2_ADMM_iterations_bla/" #print output of ADMM iterations there
                SRR_ADMM_iterations_output_filename_prefix = "TV-L2"

            ## Save text: Write header
            now = datetime.datetime.now()
            
            
            if SRR_approach in ["TK0", "TK1"]:
                # filename = self._filename_prefix
                filename = ""
                filename += SRR_approach + "-Regularization"
                filename += "_itermax" + str(SRR_iter_max)
                filename += "_" + str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2)
                filename += "_" + str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

                text = "## " + SRR_approach + "-Regularization"
                text += ", itermax = " + str(SRR_iter_max)
                text += ", tolerance = " + str(SRR_tolerance)
                text += " (" + str(now.day).zfill(2) + "." + str(now.month).zfill(2) + "." + str(now.year) 
                text += ", " + str(now.hour).zfill(2) + ":" + str(now.minute).zfill(2) + ":" + str(now.second).zfill(2) + ")"
                text += "\n## " + "alpha" + "\t" + "Residual"+ "\t" + "Psi"
                text += "\n"
            
            elif SRR_approach in ["TV-L2"]:
                # filename = self._filename_prefix
                filename = ""
                filename += SRR_approach + "-Regularization"
                filename += "_rho" + str(SRR_rho)
                filename += "_ADMM_iterations" + str(SRR_ADMM_iterations)
                filename += "_TK1itermax" + str(SRR_iter_max)
                filename += "_" + str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2)
                filename += "_" + str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

                text = "## " + SRR_approach + "-Regularization"
                text += " with " + str(SRR_ADMM_iterations) + " ADMM iterations."
                text += " TK1-solver settings:" 
                text += " itermax = " + str(SRR_iter_max)
                text += ", tolerance = " + str(SRR_tolerance)
                text += " (" + str(now.day).zfill(2) + "." + str(now.month).zfill(2) + "." + str(now.year) 
                text += ", " + str(now.hour).zfill(2) + ":" + str(now.minute).zfill(2) + ":" + str(now.second).zfill(2) + ")"
                text += "\n## " + "rho" + "\t" + "alpha" + "\t" + "Residual"+ "\t" + "Psi"
                text += "\n"

            file_handle = open(self._dir_results + filename + ".txt", "w")
            file_handle.write(text)
            file_handle.close()


            ## Run 
            for i in range(0, N_alphas):
                print("\n\t--- %s-Regularization (%s/%s): Reconstruction with alpha = %s (%s/%s) ---" %(SRR_approach, i_reg_type+1, N_regularization_types, self._alphas[i], i+1, N_alphas))
                alpha = self._alphas[i]
                HR_volume_init = st.Stack.from_stack(self._HR_volume)

                ## Super-Resolution Reconstruction object
                SRR = ips.InverseProblemSolver(self._stacks, HR_volume_init)

                ## Set current alpha
                SRR.set_alpha(alpha)

                ## Set other values
                SRR.set_alpha_cut(SRR_alpha_cut)
                
                SRR.set_iter_max(SRR_iter_max)
                SRR.set_regularization_type(SRR_approach)
                SRR.set_DTD_computation_type(SRR_DTD_computation_type)
                SRR.set_tolerance(SRR_tolerance)

                SRR.set_rho(SRR_rho)
                SRR.set_ADMM_iterations(SRR_ADMM_iterations)
                SRR.set_ADMM_iterations_output_dir(SRR_ADMM_iterations_output_dir)
                SRR.set_ADMM_iterations_output_filename_prefix(SRR_ADMM_iterations_output_filename_prefix)
                
                ## Reconstruct volume for given alpha
                SRR.run_reconstruction()
                HR_volume_alpha = SRR.get_HR_volume()

                ## Write reconstructed nifti-volume
                if save_flag:
                    filename_image =  self._filename_prefix
                    filename_image += SRR_approach
                
                    if SRR_approach in ["TK0", "TK1"]:
                        filename_image += "_itermax" + str(SRR_iter_max)
                        filename_image += "_alpha" + str(alpha)

                    elif SRR_approach in ["TV-L2"]:
                        filename_image += "_alpha" + str(alpha)
                        filename_image += "_rho" + str(SRR_rho)
                        filename_image += "_ADMM_iterations" + str(SRR_ADMM_iterations)
                        filename_image += "_TK1itermax" + str(SRR_iter_max)

                    HR_volume_alpha.write(directory=self._dir_results, filename=filename_image, write_mask=False)

                ## Compute prior term \f$ Psi(x) \f$
                self._Psis[i_reg_type, i] = self._compute_prior_term[SRR_approach](HR_volume_alpha)

                ## Compute residual \f$ \sum_k \Vert M_k y_k - M_k A_k x \Vert_{\ell^2}^2 \f$
                self._residuals[i_reg_type, i] = self._compute_residual(HR_volume_alpha)

                ## Write L-curve information into file
                file_handle = open(self._dir_results + filename + ".txt", "a")
                if SRR_approach in ["TK0", "TK1"]:
                    array_out = np.array([alpha, self._residuals[i_reg_type, i], self._Psis[i_reg_type, i]]).reshape(1,3)
                elif SRR_approach in ["TV-L2"]:
                    array_out = np.array([SRR_rho, alpha, self._residuals[i_reg_type, i], self._Psis[i_reg_type, i]]).reshape(1,4)
                
                np.savetxt(file_handle, array_out, fmt="%.10e", delimiter="\t")
                file_handle.close()
                # np.savetxt(file_handle, array_out, fmt="%.10e", delimiter="\t", header="TK0-Regulrization\nalpha\tPsi\tResidual", comments="## ")


        # self.analyse()


    def analyse(self, save_flag=0, from_files=None):   

        plot_colours = ["rx" , "bo" , "gs"]

        plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')

        ## Plot
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(1,1,1)

        if from_files is None:
            for i_reg_type in range(0, len(self._regularization_types)):

                SRR_approach = self._regularization_types[i_reg_type]

                print("\t --- %s Regularization ---" %(SRR_approach))
                if SRR_approach in ["TK0", "TK1"]:
                    ## Print on screen
                    print("\t\talpha\t\tResidual\t\tPrior term Psi")
                    for i in range(0, len(self._alphas)):
                        print("\t\t%s\t\t%.3e\t\t%.3e" %(self._alphas[i], self._residuals[i_reg_type, i], self._Psis[i_reg_type, i]))
                elif SRR_approach in ["TV-L2"]:
                    ## Print on screen
                    print("\t\trho\t\talpha\t\tResidual\t\tPrior term Psi")
                    for i in range(0, len(self._alphas)):
                        print("\t\t%s\t\t%.3e\t\t%.3e" %(self._rho, self._alphas[i], self._residuals[i_reg_type, i], self._Psis[i_reg_type, i]))


                ## Plot curve
                ax.plot(self._residuals[i_reg_type,:], self._Psis[i_reg_type,:], plot_colours[i_reg_type], label=SRR_approach)
                # ax.loglog(self._residuals[i_reg_type,:], self._Psis[i_reg_type,:], plot_colours[i_reg_type], label=self._regularization_types[i_reg_type])

        else:
            for i_file in range(0, len(from_files)):
                # data = np.loadtxt(from_files[i_file] + ".txt" , delimiter="\t", skiprows=2)
                data = np.loadtxt(from_files[i_file] + ".txt" , skiprows=2)

                if "TK0-Regularization" in from_files[i_file]:
                    SRR_approach = "TK0"
                    alphas = data[:,0]
                    residuals = data[:,1]
                    Psis = data[:,2]

                elif "TK1-Regularization" in from_files[i_file]:
                    SRR_approach = "TK1"
                    alphas = data[:,0]
                    residuals = data[:,1]
                    Psis = data[:,2]

                elif "TV-L2-Regularization" in from_files[i_file]:
                    SRR_approach = "TV-L2"
                    rhos = data[:,0]
                    alphas = data[:,1]
                    residuals = data[:,2]
                    Psis = data[:,3]

                else:
                    raise ValueError("Error: SRR method could not be determined from filename")
                # SRR_approach = re.sub("-Regularization.*","",from_files[i_file])
                # SRR_approach = re.sub(".*_","",SRR_approach)


                # scale = data[:,2].mean()
                # residuals /= scale
                # Psis /= scale

                ## Print on screen
                print("\t --- %s-Regularization ---" %(SRR_approach))
                if SRR_approach in ["TK0", "TK1"]:
                    label=SRR_approach
                    print("\t\talpha\t\tResidual\t\tPrior term Psi")
                    for i in range(0, len(alphas)):
                        print("\t\t%s\t\t%.3e\t\t%.3e" %(alphas[i], residuals[i], Psis[i]))
                elif SRR_approach in ["TV-L2"]:
                    label=SRR_approach+" (rho=" + str(rhos[0]) + ")"
                    print("\t\trho\t\talpha\t\tResidual\t\tPrior term Psi")
                    for i in range(0, len(alphas)):
                        print("\t\t%s\t\t%s\t\t%.3e\t\t%.3e" %(rhos[i], alphas[i], residuals[i], Psis[i]))
                ## Plot curve
                ax.plot(residuals, Psis, plot_colours[i_file], label=label)
                # ax.loglog(residuals[i_file], Psis[i_file], plot_colours[i_file], label=SRR_approach)

                ## Maximum curvature
                # i_max_curvature = self._get_maximum_curvature_point(residuals, Psis)
                # ax.plot(residuals[i_max_curvature], Psis[i_max_curvature], "cs")

                ## Fit curve
                # bounds = (0, [None, None, None])
                # popt, pcov = curve_fit(self._fitting_curve, residuals, Psis)
                # xmin = np.min(residuals)
                # xmax = np.max(residuals)
                # step = (xmax-xmin)/50
                # xdata = np.arange(xmin, xmax, step)

                # # print popt
                # ax.plot(xdata, self._fitting_curve(xdata, *popt))


        ## Show grid
        ax.grid()

        ## Add legend
        legend = ax.legend(loc="upper right")

        plt.title("L-curve\n$\Phi(\\vec{x}) = \\frac{1}{2}\sum_{k=1}^K \Vert M_k (\\vec{y}_k - A_k \\vec{x}) \Vert_{\ell^2}^2 + \\alpha \Psi(\\vec{x})\\rightarrow \min$" )
        # plt.title("L-curve for " + regularization_types[i_reg_type] + "Regularization")
        plt.ylabel("Prior term $\displaystyle\Psi(\\vec{x}_\\alpha)$")
        plt.xlabel("Residual $\sum_{k=1}^K \Vert M_k (\\vec{y}_k - A_k \\vec{x}_\\alpha) \Vert_{\ell^2}^2$")

        plt.draw()
        plt.pause(0.5) ## important! otherwise fig is not shown. Also needs plt.show() at the end of the file to keep figure open

        if save_flag:
            fig.savefig(self._dir_results + "L-curve.eps")


    def _fitting_curve(self, x, a,b,c):

        # foo = np.polyval([a,b,c,d],x)
        foo = a*np.exp(-b*x) + c

        return foo

    def _get_maximum_curvature_point(self, x, y):

        # scale = x.mean()
        scale = 1

        x = x/scale
        y = y/scale

        N_points = len(x)
        radius2 = np.zeros(N_points-2)

        M = np.zeros((3,3))

        for i in range(1, N_points-1):
            M[:,0] = 1
            M[:,1] = x[i-1:i+2]
            M[:,2] = y[i-1:i+2]
            b = x[i-1:i+2]**2 + y[i-1:i+2]**2

            [A, B, C] = np.linalg.solve(M,b)

            radius2[i-1] = A + (B**2 + C**2)/4
            print("(xm, ym, r2) = (%s, %s, %s)" %(B/2.,C/2.,radius2[i-1]))

        i_max_curvature = np.argmin(radius2)+1

        return i_max_curvature



    def _compute_prior_term_TK0(self, HR_volume):
        ## Get data array
        nda = self._itk2np.GetArrayFromImage( HR_volume.itk )

        return np.linalg.norm(nda)**2


    def _compute_prior_term_TK1(self, HR_volume):
        ## Get data array
        nda = self._itk2np.GetArrayFromImage( HR_volume.itk )

        spacing = HR_volume.sitk.GetSpacing()

        ## Get kernels for differentiation and isotropic spacing
        kernel_Dx = self._get_forward_diff_x_kernel() / spacing[0]
        kernel_Dy = self._get_forward_diff_y_kernel() / spacing[0]
        kernel_Dz = self._get_forward_diff_z_kernel() / spacing[0]

        ## Differentiate
        Dx = self._convolve(nda, kernel_Dx)
        Dy = self._convolve(nda, kernel_Dy)
        Dz = self._convolve(nda, kernel_Dz)

        ## Compute norm || Dx ||^2 with D = [D_x; D_y; D_z]
        return np.linalg.norm(Dx)**2 + np.linalg.norm(Dy)**2 + np.linalg.norm(Dz)**2

    ## TODO
    def _compute_prior_term_TVL2(self, HR_volume):
        ## Get data array
        nda = self._itk2np.GetArrayFromImage( HR_volume.itk )

        spacing = HR_volume.sitk.GetSpacing()

        ## Get kernels for differentiation and isotropic spacing
        kernel_Dx = self._get_forward_diff_x_kernel() / spacing[0]
        kernel_Dy = self._get_forward_diff_y_kernel() / spacing[0]
        kernel_Dz = self._get_forward_diff_z_kernel() / spacing[0]

        ## Compute finite differences
        Dx = self._convolve(nda, kernel_Dx)
        Dy = self._convolve(nda, kernel_Dy)
        Dz = self._convolve(nda, kernel_Dz)

        ## Compute TV(x) = ||Dx||_{2,1} with D = [D_x; D_y; D_z]
        return np.sum(np.sqrt(Dx**2 + Dy**2 + Dz**2))

    ## Compute residual
    # \f[
    #   \sum_k \Vert M_k y_k - M_k A_k x \Vert_{\ell^2}^2
    # \f]
    # \param[in] HR_volume
    def _compute_residual(self, HR_volume):

        residual = 0.;

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            for j in range(0, stack.get_number_of_slices()):

                slice = slices[j]
                # sitkh.show_itk_image(slice.itk,title="slice")

                ## Add contribution || M_k (y_k - A_k x) ||^2
                residual += self._get_squared_norm_M_y_minus_Ax(slice, HR_volume)

        return residual


    def _get_squared_norm_M_y_minus_Ax(self, slice, HR_volume):

        ## TODO: Write itkBinaryFunctorImageFilter to multiply quicker
        ## Also, by changing to 
        ## itk.MultiplyImageFilter[itk.Image[itk.UC,3], image_type, image_type].New()
        ## the mask could be considered a different type, see sitkh.convert_sitk_to_itk_image
        ## (which in turn could help mask-based registration with ITK objects)
        # multiplier = itk.MultiplyImageFilter[itk.Image[itk.UC,3], image_type, image_type].New()
        multiplier = itk.MultiplyImageFilter[image_type, image_type, image_type].New()

        ## Update A_k based on relative position of slice
        self._update_oriented_gaussian_image_filter(slice, HR_volume)

        ## Compute A_k x
        Ax = self._A(HR_volume.itk)

        ## Compute y_k - A_k x
        y_minus_Ax = self._add_amplified_image(slice.itk, -1, Ax)

        ## Compute M_k (y_k - A_k x)
        multiplier.SetInput1( slice.itk_mask )
        multiplier.SetInput2( y_minus_Ax )
        multiplier.Update()

        M_y_minus_Ax = multiplier.GetOutput()
        M_y_minus_Ax.DisconnectPipeline()

        ## Compute || M_k (y_k - A_k x) ||^2
        nda = self._itk2np.GetArrayFromImage(M_y_minus_Ax)

        return np.linalg.norm(nda)**2


    ## Update internal Oriented Gaussian Interpolate Image Filter parameters. 
    #  Hence, update combined Downsample and Blur Operator according to the 
    #  relative position between slice and HR volume.
    #  \param[in] slice Slice object
    #  \param[in] HR_volume Stack object
    def _update_oriented_gaussian_image_filter(self, slice, HR_volume):
        ## Get variance covariance matrix representing Gaussian blurring in HR volume coordinates
        Cov_HR_coord = self._psf.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates( slice, HR_volume )

        ## Update parameters of forward operator A
        self._filter_oriented_Gaussian_interpolator.SetCovariance( Cov_HR_coord.flatten() )
        self._filter_oriented_Gaussian.SetOutputParametersFromImage( slice.itk )
        

    ## Perform forward operation on HR image, i.e. \f$y = DBx =: Ax \f$ with \f$D\f$  and \f$ B \f$ being 
    #  the downsampling and blurring operator, respectively.
    #  \param[in] HR_volume_itk HR image as itk.Image object
    #  \return image in LR space as itk.Image object after performed forward operation
    def _A(self, HR_volume_itk):
        HR_volume_itk.Update()
        self._filter_oriented_Gaussian.SetInput( HR_volume_itk )
        self._filter_oriented_Gaussian.UpdateLargestPossibleRegion()
        self._filter_oriented_Gaussian.Update()

        slice_itk = self._filter_oriented_Gaussian.GetOutput();
        slice_itk.DisconnectPipeline()

        return slice_itk


    ## Compute I0 + const*I1 with I0 and I1 being itk.Image objects occupying
    #  the same physical space
    #  \param[in] image0_itk first image, itk.Image object
    #  \param[in] const constant to multiply second image, scalar
    #  \param[in] image1_itk second image, itk.Image object
    #  \return image0_itk + const*image1_itk as itk.Image object
    def _add_amplified_image(self, image0_itk, const, image1_itk):

        ## Create image adder and multiplier
        adder = itk.AddImageFilter[image_type, image_type, image_type].New()
        multiplier = itk.MultiplyImageFilter[image_type, image_type, image_type].New()

        ## compute const*image1_itk
        multiplier.SetInput( image1_itk )
        multiplier.SetConstant( const )

        ## compute image0_itk + const*image1_itk
        adder.SetInput1( image0_itk )
        adder.SetInput2( multiplier.GetOutput() )
        adder.Update()

        res = adder.GetOutput()
        res.DisconnectPipeline()

        return res


    """
    TK1-regularization functions
    """
    ## Compute forward difference quotient in x-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(nda) to differentiate image
    #  \return kernel for 3-dimensional differentiation in x
    def _get_forward_diff_x_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,1,2))
        kernel[:] = np.array([1,-1])

        return kernel


    ## Compute backward difference quotient in x-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(nda) to differentiate image
    #  \return kernel for 3-dimensional differentiation in x
    def _get_backward_diff_x_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,1,3))
        kernel[:] = np.array([0,1,-1])

        return kernel


    ## Compute forward difference quotient in y-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(kernel, nda) to differentiate image
    #  \return kernel kernel for 3-dimensional differentiation in y
    def _get_forward_diff_y_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,2,1))
        kernel[:] = np.array([[1],[-1]])

        return kernel


    ## Compute backward difference quotient in y-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(kernel, nda) to differentiate image
    #  \return kernel kernel for 3-dimensional differentiation in y
    def _get_backward_diff_y_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((1,3,1))
        kernel[:] = np.array([[0],[1],[-1]])

        return kernel


    ## Compute forward difference quotient in z-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(kernel, nda) to differentiate image
    #  \return kernel kernel for 3-dimensional differentiation in z
    def _get_forward_diff_z_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((2,1,1))
        kernel[:] = np.array([[[1]],[[-1]]])

        return kernel


    ## Compute backward difference quotient in y-direction to differentiate
    #  array with array = array[z,y,x], i.e. the 'correct' direction
    #  by viewing the resulting nifti-image differentiation. The resulting kernel
    #  can be used via _convolve(kernel, nda) to differentiate image
    #  \return kernel kernel for 3-dimensional differentiation in z
    def _get_backward_diff_z_kernel(self):
        ## kernel = np.zeros((z,y,x))
        kernel = np.zeros((3,1,1))
        kernel[:] = np.array([[[0]],[[1]],[[-1]]])

        return kernel


    ## Compute convolution of array based on given kernel via 
    #  scipy.ndimage.convolve with "wrap" boundary conditions.
    #  \param[in] nda data array
    #  \param[in] kernel 
    #  \return data array convolved by given kernel
    def _convolve(self, nda, kernel):
        return ndimage.convolve(nda, kernel, mode='wrap')