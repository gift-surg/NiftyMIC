#!/usr/bin/python

## \file TVL2Solver.py
#  \brief Implementation to get an approximate solution of the inverse problem 
#  \f$ y_k = A_k x \f$ for each slice \f$ y_k,\,k=1,\dots,K \f$
#  by using TV-L2-regularization. 
#  Solution via Alternating Direction Method of Multipliers (ADMM) method.
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date July 2016

## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear
from scipy.optimize import nnls
import time
from datetime import timedelta

## Import modules
import utilities.SimpleITKHelper as sitkh
import reconstruction.solver.TikhonovSolver as tk
from reconstruction.solver.Solver import Solver


## This class implements the framework to iteratively solve 
#  \f$ \vec{y}_k = A_k \vec{x} \f$ for every slice \f$ \vec{y}_k,\,k=1,\dots,K \f$
#  via TV-L2-regularization via an augmented least-square approach
#  TODO
class TVL2Solver(Solver):

    ##
    #          Constructor
    # \date          2016-08-01 22:57:21+0100
    #
    # \param         self                                    The object
    # \param[in]     stacks                                  list of Stack
    #                                                        objects containing
    #                                                        all stacks used
    #                                                        for the
    #                                                        reconstruction
    # \param[in,out] HR_volume                               Stack object
    #                                                        containing the
    #                                                        current estimate
    #                                                        of the HR volume
    #                                                        (used as initial
    #                                                        value + space
    #                                                        definition)
    # \param[in]     alpha_cut                               Cut-off distance
    #                                                        for Gaussian
    #                                                        blurring filter
    # \param[in]     alpha                                   regularization
    #                                                        parameter, scalar
    # \param[in]     iter_max                                number of maximum
    #                                                        iterations, scalar
    # \param[in]     rho                                     regularization
    #                                                        parameter of
    #                                                        augmented
    #                                                        Lagrangian term,
    #                                                        scalar
    # \param[in]     ADMM_iterations                         number of ADMM
    #                                                        iterations, scalar
    # \param[in]     ADMM_iterations_output_dir              The ADMM iterations output dir
    # \param[in]     ADMM_iterations_output_filename_prefix  The ADMM iterations output filename prefix
    #
    def __init__(self, stacks, HR_volume, alpha_cut=3, alpha=0.03, iter_max=10, minimizer="lsmr", deconvolution_mode="full_3D", predefined_covariance=None, rho=0.5, ADMM_iterations=10, ADMM_iterations_output_dir=None, ADMM_iterations_output_filename_prefix="TVL2"):

        ## Run constructor of superclass
        Solver.__init__(self, stacks=stacks, HR_volume=HR_volume, alpha_cut=alpha_cut, alpha=alpha, iter_max=iter_max, minimizer=minimizer, deconvolution_mode=deconvolution_mode, predefined_covariance=predefined_covariance)               
        
        ## Settings for optimizer
        self._rho = rho
        self._ADMM_iterations = ADMM_iterations

        self._ADMM_iterations_output_dir = ADMM_iterations_output_dir
        self._ADMM_iterations_output_filename_prefix = ADMM_iterations_output_filename_prefix


    ## Set regularization parameter used for augmented Lagrangian in TV-L2 regularization
    #  \[$
    #   \sum_{k=1}^K \frac{1}{2} \Vert y_k - A_k x \Vert_{\ell^2}^2 + \alpha\,\Psi(x) 
    #   + \mu \cdot (\nabla x - v) + \frac{\rho}{2} \Vert \nabla x - v \Vert_{\ell^2}^2
    #  \]$
    #  \param[in] rho regularization parameter of augmented Lagrangian term, scalar
    def set_rho(self, rho):
        self._rho = rho


    ## Get regularization parameter used for augmented Lagrangian in TV-L2 regularization
    #  \return regularization parameter of augmented Lagrangian term, scalar
    def get_rho(self):
        return self._rho


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


    ## Set ouput directory to write TV results in case outputs of ADMM iterations are desired
    #  \param[in] dir_output directory to write TV results, string
    def set_ADMM_iterations_output_dir(self, dir_output):
        self._ADMM_iterations_output_dir = dir_output


    ## Get ouput directory to write TV results in case outputs of ADMM iterations are desired
    def get_ADMM_iterations_output_dir(self):
        return self._ADMM_iterations_output_dir


    ## Set filename prefix to write TV reconstructed volumes of ADMM iteration results 
    #  \pre ADMM_iterations_output_dir was set
    #  \param[in] filename filename prefix of ADMM output iteration volumes, string
    def set_ADMM_iterations_output_filename_prefix(self, filename):
        self._ADMM_iterations_output_filename_prefix = filename


    ## Get filename to write TV reconstructed volumes of ADMM iteration results 
    #  \pre ADMM_iterations_output_dir was set
    def get_ADMM_iterations_output_filename_prefix(self, filename):
        return self._ADMM_iterations_output_filename_prefix


    ## Compute statistics associated to performed reconstruction
    def compute_statistics(self):
        HR_nda_vec = sitk.GetArrayFromImage(self._HR_volume.sitk).flatten()

        self._residual_ell2 = self._get_residual_ell2(HR_nda_vec)
        self._residual_prior = self._get_residual_prior(HR_nda_vec)


    ## Print statistics associated to performed reconstruction
    def print_statistics(self):
        print("\nStatistics for performed reconstruction with TV-L2-regularization:")
        # if self._elapsed_time_sec < 0:
        #     raise ValueError("Error: Elapsed time has not been measured. Run 'run_reconstruction' first.")
        # else:
        print("\tElapsed time: %s" %(timedelta(seconds=self._elapsed_time_sec)))
        print("\tell^2-residual sum_k ||M_k(A_k x - y_k)||_2^2 = %.3e" %(self._residual_ell2))
        print("\tprior residual = %.3e" %(self._residual_prior))


    ##
    #       Gets the setting specific filename indicating the information
    #             used for the reconstruction step
    # \date       2016-11-17 15:41:58+0000
    #
    # \param      self    The object
    # \param      prefix  The prefix as string
    #
    # \return     The setting specific filename as string.
    #
    def get_setting_specific_filename(self, prefix="recon_"):
        
        ## Build filename
        filename = prefix
        filename += "stacks" + str(len(self._stacks))
        if self._alpha > 0:
            filename += "_TVL2"
        filename += "_" + self._minimizer
        filename += "_alpha" + str(self._alpha)
        filename += "_itermax" + str(self._iter_max)
        filename += "_rho" + str(self._rho)
        filename += "_ADMMiterations" + str(self._ADMM_iterations)
        
        ## Replace dots by 'p'
        filename = filename.replace(".","p")

        return filename


    ##
    #       Reconstruct volume using TV-L2 regularization via Alternating
    #             Direction Method of Multipliers (ADMM) method.
    # \post       self._HR_volume is updated with new volume and can be fetched
    #             via \p get_HR_volume
    # \date       2016-08-01 23:22:50+0100
    #
    # \param      self                    The object
    # \param      estimate_initial_value  Estimate initial value by running one
    #                                     first-order Tikhonov reconstruction
    #                                     step prior the ADMM algorithm
    #
    def run_reconstruction(self, estimate_initial_value=True):

        print("Chosen regularization type: TV-L2")
        print("Regularization parameter alpha: " + str(self._alpha))
        print("Regularization parameter of augmented Lagrangian term rho: " + str(self._rho))
        print("Number of ADMM iterations: " + str(self._ADMM_iterations))
        print("Maximum number of TK1 solver iterations: " + str(self._iter_max))
        if self._ADMM_iterations_output_dir is not None:
            print("ADMM iterations are written to " + self._ADMM_iterations_output_dir)
        # print("Tolerance: %.0e" %(self._tolerance))

        time_start = time.time()

        ## Estimate initial value based on TK1 regularization
        if estimate_initial_value:
            print("\n***********************************************************************************")
            print("Initial volume for ADMM is estimated by prior TK1 reconstruction step")
            self._compute_initial_value_based_on_TK1(alpha_cut=3, alpha=0.02, iter_max=10, minimizer="lsmr",deconvolution_mode=self._deconvolution_mode, predefined_covariance=self._predefined_covariance)

        ## Get data array of current volume estimate
        HR_nda = sitk.GetArrayFromImage(self._HR_volume.sitk)

        ##  Set initial values
        vx_nda = self._differential_operations.Dx(HR_nda) # differential in x
        vy_nda = self._differential_operations.Dy(HR_nda) # differential in y
        vz_nda = self._differential_operations.Dz(HR_nda) # differential in z

        wx_nda = np.zeros_like(vx_nda)
        wy_nda = np.zeros_like(vy_nda)
        wz_nda = np.zeros_like(vz_nda)
    
        ## Fill My -part of b, i.e. masked slices stacked to 1D vector 
        #  (remaining elements will be updated at every iteration)
        b = np.zeros(self._N_total_slice_voxels + 3*self._N_voxels_HR_volume)
        b[0:self._N_total_slice_voxels] = self._get_M_y()

        ## Perform ADMM_iterations
        for iter in range(0, self._ADMM_iterations):
            print("\n***********************************************************************************")
            print("ADMM iteration %s/%s:" %(iter+1, self._ADMM_iterations))

            ## Perform ADMM step
            HR_nda, vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda = self._perform_ADMM_iteration(HR_nda, b, vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda, self._alpha, self._rho)

            if self._ADMM_iterations_output_dir is not None:
                
                ## Only use signifcant digits for string
                alpha_str = "%g" % self._alpha
                rho_str = "%g" % self._rho

                ## Build filename
                name =  self._ADMM_iterations_output_filename_prefix
                name += "_stacks" + str(self._N_stacks)
                name += "_rho" + rho_str
                name += "_alpha" + alpha_str
                name += "_TK1itermax" + str(self._iter_max)
                name += "_ADMMiteration" + str(iter+1)

                ## Replace decimal point by 'p'
                name = name.replace(".", "p")

                os.system("mkdir -p " + self._ADMM_iterations_output_dir)

                HR_volume_itk = self._get_itk_image_from_array_vec(HR_nda.flatten(), self._HR_volume.itk)
                sitkh.write_itk_image(HR_volume_itk, self._ADMM_iterations_output_dir+name+".nii.gz")              

            ## DEBUG:
            ## Show reconstruction
            # sitkh.show_itk_image(self._get_itk_image_from_array_vec(HR_nda.flatten(), self._HR_volume.itk), title="HR_volume_iteration_"+str(iter+1))

            ## Show auxiliary v = Dx
            # sitkh.show_itk_image(self._get_HR_image_from_array_vec(vx_nda.flatten()), title="vx_iteration_"+str(iter+1))
            # sitkh.show_itk_image(self._get_HR_image_from_array_vec(vy_nda.flatten()), title="vy_iteration_"+str(iter+1))
            # sitkh.show_itk_image(self._get_HR_image_from_array_vec(vz_nda.flatten()), title="vz_iteration_"+str(iter+1))

            ## Show scaled dual variable w
            # sitkh.show_itk_image(self._get_HR_image_from_array_vec(wx_nda.flatten()), title="wx_iteration_"+str(iter+1))
            # sitkh.show_itk_image(self._get_HR_image_from_array_vec(wy_nda.flatten()), title="wy_iteration_"+str(iter+1))
            # sitkh.show_itk_image(self._get_HR_image_from_array_vec(wz_nda.flatten()), title="wz_iteration_"+str(iter+1))

        ## Set elapsed time
        time_end = time.time()
        self._elapsed_time_sec = time_end-time_start

        ## Update volume
        self._HR_volume.itk = self._get_itk_image_from_array_vec( HR_nda.flatten(), self._HR_volume.itk )
        self._HR_volume.sitk = sitkh.get_sitk_from_itk_image( self._HR_volume.itk )


    ##
    #       Calculates the initial value based on first-order Tikhonov.
    # \post       self._HR_volume is updated
    # \date       2016-08-01 22:51:41+0100
    #
    # \param      self       The object
    # \param[in]  alpha_cut  Cut-off distance for Gaussian blurring filter
    # \param[in]  alpha      The alpha
    # \param[in]  iter_max   The iterator maximum
    #
    def _compute_initial_value_based_on_TK1(self, alpha_cut, alpha, iter_max, minimizer, deconvolution_mode, predefined_covariance):
        SRR = tk.TikhonovSolver(self._stacks, self._HR_volume, alpha_cut=alpha_cut, alpha=alpha, iter_max=iter_max, reg_type="TK1", minimizer=minimizer, deconvolution_mode=deconvolution_mode, predefined_covariance=predefined_covariance)
        SRR.run_reconstruction()


    ## Perform single ADMM iteration
    #  \param[in] HR_nda initial value of HR volume data array, numpy array
    #  \param[in] b right-hand side \f$ b = \begin{pmatrix} M_1 \vec{y}_1 \\ M_2 \vec{y}_2 \\ \vdots \\ M_K \vec{y}_K \\ \sqrt{\rho}(\vec{v}^i-\vec{w}^i)\end{pmatrix} \f$ 
    #  \param[in] vx_nda initial value for auxiliary variable for decoupled 
    #       but constrained primal problem, i.e.
    #       i.e. \f$ v = Df \f$ with \f$f\f$ being the solution term,
    #       in x-direction, numpy array
    #  \param[in] vy_nda initial value for auxiliary variable in y-direction
    #  \param[in] vz_nda initial value for auxiliary variable in z-direction
    #  \param[in] wx_nda initial value for scaled dual variable (by \p rho)
    #       originating from augmented Lagrangian in x-direction, numpy array
    #  \param[in] wy_nda initial value for scaled dual variable in y-direction
    #  \param[in] wz_nda initial value for scaled dual variable in z-direction
    #  \param[in] alpha regularization parameter for primal problem, scaler>0
    #  \param[in] rho regularization parameter for augmented Lagrangian term, scalar>0
    #  \return updated HR volume data array
    #  \return updated auxiliary variables \p v in x-, y- and z-direction
    #  \return updated scaled dual variable \p w in x-, y- and z-direction
    def _perform_ADMM_iteration(self, HR_nda, b, vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda, alpha, rho):

        ## 1) Solve for x^{k+1} by using first-order Tikhonov regularization
        HR_nda = self._perform_ADMM_step_1_TK1_recon_solution(HR_nda, b, vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda, rho)

        ## Compute derivatives for subsequent steps
        Dx_nda = self._differential_operations.Dx(HR_nda) 
        Dy_nda = self._differential_operations.Dy(HR_nda) 
        Dz_nda = self._differential_operations.Dz(HR_nda) 

        ## 2) Solve for v^{k+1}
        vx_nda, vy_nda, vz_nda = self._perform_ADMM_step_2_auxiliary_variable_v(Dx_nda, Dy_nda, Dz_nda, wx_nda, wy_nda, wz_nda, alpha/rho)

        ## 3) Solve for w^{k+1}, i.e. scaled dual variable
        wx_nda, wy_nda, wz_nda = self._perform_ADMM_step_3_scaled_dual_variable(Dx_nda, Dy_nda, Dz_nda, vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda)

        return HR_nda, vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda


    ## Perform first step of ADMM algorithm:
    # TODO
    #  \param[in] HR_nda initial value of HR volume data array, numpy array
    #  \param[in] b right-hand side \f$ b = \begin{pmatrix} M_1 \vec{y}_1 \\ M_2 \vec{y}_2 \\ \vdots \\ M_K \vec{y}_K \\ \sqrt{rho}(\vec{v}^i-\vec{w}^i)\end{pmatrix} \f$ 
    #  \param[in] vx_nda initial value for auxiliary variable for decoupled 
    #       but constrained primal problem, i.e.
    #       i.e. \f$ v = Df \f$ with \f$f\f$ being the solution term,
    #       in x-direction, numpy array
    #  \param[in] vy_nda initial value for auxiliary variable in y-direction
    #  \param[in] vz_nda initial value for auxiliary variable in z-direction
    #  \param[in] wx_nda initial value for scaled dual variable (by \p rho)
    #       originating from augmented Lagrangian in x-direction, numpy array
    #  \param[in] wy_nda initial value for scaled dual variable in y-direction
    #  \param[in] wz_nda initial value for scaled dual variable in z-direction
    #  \param[in] rho regularization parameter for augmented Lagrangian term, scalar>0
    #  \return updated HR volume data array
    def _perform_ADMM_step_1_TK1_recon_solution(self, HR_nda, b, vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda, rho):
        print("TK1-regularization step" )
        print("\tMaximum number of TK1 solver iterations: " + str(self._iter_max))
        print("\tRegularization parameter alpha: " + str(self._alpha))
        print("\tRegularization parameter of augmented Lagrangian term rho: " + str(self._rho))
        if self._deconvolution_mode in ["only_in_plane"]:
            print("\t(Only in-plane deconvolution is performed)")

        elif self._deconvolution_mode in ["predefined_covariance"]:
            print("\t(Predefined covariance used: cov = diag(%s))" % (np.diag(self._predefined_covariance)))
        # print("\tTolerance: %.0e" %(self._tolerance))

        ## Update changing elements of right-hand side b
        b[-3*self._N_voxels_HR_volume:] = np.sqrt(rho)*self._get_unscaled_b_lower(vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda)

        ## Construct (sparse) linear operator A
        A_fw = lambda x: self._A_TK1(x, rho)
        A_bw = lambda x: self._A_adj_TK1(x, rho)

        ## Run solver to compute least-square solution based on TK1-regularization
        HR_nda_vec = self._get_approximate_solution[self._minimizer](A_fw, A_bw, b)

        ## Extract estimated solution as numpy array
        HR_nda = HR_nda_vec.reshape(self._HR_shape_nda)
        
        return HR_nda


    ## Compute unscaled lower part of b, i.e.
    #  \f$ \vec{v}^i - \vec{w}^i\f$ 
    #  \param[in] N_voxels number of voxels (only two possibilities depending on G), integer
    #  \return vector lower part of b as 1D array
    def _get_unscaled_b_lower(self, vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda):

        ## Allocate memory
        b = np.zeros(3*self._N_voxels_HR_volume)

        ## Compute changing, lower part of b without scaling
        b[-3*self._N_voxels_HR_volume:-2*self._N_voxels_HR_volume] = (vx_nda - wx_nda).flatten()
        b[-2*self._N_voxels_HR_volume:-self._N_voxels_HR_volume] = (vy_nda - wy_nda).flatten()
        b[-self._N_voxels_HR_volume:] = (vz_nda - wz_nda).flatten()

        return b


    ## Perform second step of ADMM algorithm:
    # TODO
    #  \param[in] Dx_nda Derivative of HR volume data array in x-direction, numpy array
    #  \param[in] Dy_nda Derivative of HR volume data array in y-direction, numpy array
    #  \param[in] Dz_nda Derivative of HR volume data array in z-direction, numpy array
    #  \param[in] wx_nda initial value for scaled dual variable (by \p rho)
    #       originating from augmented Lagrangian in x-direction, numpy array
    #  \param[in] wy_nda initial value for scaled dual variable in y-direction
    #  \param[in] wz_nda initial value for scaled dual variable in z-direction
    #  \param[in] ell scaled regularization variable, i.e. alpha/rho, used for threshold
    #  \return Updates of auxiliary variable v in x-, y- and z-direction
    def _perform_ADMM_step_2_auxiliary_variable_v(self, Dx_nda, Dy_nda, Dz_nda, wx_nda, wy_nda, wz_nda, ell):

        ## Compute t = Dx + w
        tx_nda = Dx_nda + wx_nda
        ty_nda = Dy_nda + wy_nda
        tz_nda = Dz_nda + wz_nda

        ## Compute auxiliary variable for decoupled but constrained problem
        return self._vectorial_soft_threshold(ell, tx_nda, ty_nda, tz_nda)

    
    ## Perform third step of ADMM algorithm:
    #  \param[in] Dx_nda Derivative of HR volume data array in x-direction, numpy array
    #  \param[in] Dy_nda Derivative of HR volume data array in y-direction, numpy array
    #  \param[in] Dz_nda Derivative of HR volume data array in z-direction, numpy array
    #  \param[in] vx_nda initial value for auxiliary variable for decoupled 
    #       but constrained primal problem, i.e.
    #       i.e. \f$ v = Df \f$ with \f$f\f$ being the solution term,
    #       in x-direction, numpy array
    #  \param[in] vy_nda initial value for auxiliary variable in y-direction
    #  \param[in] vz_nda initial value for auxiliary variable in z-direction
    #  \param[in] wx_nda initial value for scaled dual variable (by \p rho)
    #       originating from augmented Lagrangian in x-direction, numpy array
    #  \param[in] wy_nda initial value for scaled dual variable in y-direction
    #  \param[in] wz_nda initial value for scaled dual variable in z-direction
    #  \return Updates of scaled dual variable w in x-, y- and z-direction
    def _perform_ADMM_step_3_scaled_dual_variable(self, Dx_nda, Dy_nda, Dz_nda, vx_nda, vy_nda, vz_nda, wx_nda, wy_nda, wz_nda):

        ## Compute w_new = w + Dx - v
        wx_nda = wx_nda + Dx_nda - vx_nda
        wy_nda = wy_nda + Dy_nda - vy_nda
        wz_nda = wz_nda + Dz_nda - vz_nda

        return wx_nda, wy_nda, wz_nda


    ## Get vectorial soft threshold based on \p get_soft_threshold \f$ S_\ell \f$
    #  as solution of TODO
    #  \param[in] ell scalar > 0 defining the threshold
    #  \param[in] tx_nda x-coordinate of substituted variable Dx + w as numpy array 
    #       \f$ \in \mathbb{R}^{\text{dim}_x\times \text{dim}_y} \f$
    #  \param[in] ty_nda y-coordinate of substituted variable Dx + w as numpy array 
    #  \param[in] tz_nda z-coordinate of substituted variable Dx + w as numpy array 
    #  \return vectorial soft threshold
    #       \f[ \vec{S_\ell}(\vec{t}) = \begin{cases}
    #           \frac{S_\ell(\Vert \vec{t} \Vert_2)}{\Vert \vec{t} \Vert_2} \vec{t}
    #               , & \text{if } \Vert \vec{t} \Vert_2 > \ell \\
    #           0, & \text{otherwise}
    #       \end{cases}
    #       \f]
    def _vectorial_soft_threshold(self, ell, tx_nda, ty_nda, tz_nda):

        Sx = np.zeros_like(tx_nda)
        Sy = np.zeros_like(ty_nda)
        Sz = np.zeros_like(tz_nda)

        t_norm = np.sqrt(tx_nda**2 + ty_nda**2 + tz_nda**2)

        ind = t_norm > ell

        Sx[ind] = self._soft_threshold(ell, t_norm[ind])*tx_nda[ind]/t_norm[ind]
        Sy[ind] = self._soft_threshold(ell, t_norm[ind])*ty_nda[ind]/t_norm[ind]
        Sz[ind] = self._soft_threshold(ell, t_norm[ind])*tz_nda[ind]/t_norm[ind]

        return Sx, Sy, Sz


    ## Get soft threshold as solution of TODO
    #  \param[in] ell threshold as scalar > 0
    #  \param[in] t array containing the values to be thresholded
    #  \return soft threshold
    #       \f[ S_\ell(t) 
    #       =  \max(|t|-\ell,0)\,\text{sgn}(t)
    #       = \begin{cases}
    #           t-\ell,& \text{if } t>\ell \\
    #           0,& \text{if } |t|\le\ell \\
    #           t+\ell,& \text{if } t<\ell
    #       \end{cases}
    #       \f]
    def _soft_threshold(self, ell, t):
        return np.maximum(np.abs(t) - ell, 0)*np.sign(t)


    ## Compute residual for TV-L2-regularization prior, i.e.
    #  TV(x) = ||Dx||_{2,1} with D = [D_x; D_y; D_z]
    #  \param[in] HR_nda_vec HR data as 1D array
    #  \return TV(x)
    def _get_residual_prior(self, HR_nda_vec):
        HR_nda = HR_nda_vec.reshape(self._HR_shape_nda)

        Dx = self._differential_operations.Dx(HR_nda)
        Dy = self._differential_operations.Dy(HR_nda)
        Dz = self._differential_operations.Dz(HR_nda)

        ## Compute TV(x) = ||Dx||_{2,1} with D = [D_x; D_y; D_z]
        return np.sum(np.sqrt(Dx**2 + Dy**2 + Dz**2))
