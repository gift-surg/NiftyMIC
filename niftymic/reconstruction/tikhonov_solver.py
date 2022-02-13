##
# \file tikhonov_solver.py
# \brief      Implementation to get an approximate solution of the SRR
#             problem using Tikhonov-regularization
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2016
#

import scipy
import numpy as np
import SimpleITK as sitk

from nsol.definitions import EPS
import nsol.linear_operators as linop
import nsol.tikhonov_linear_solver as tk
import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from niftymic.reconstruction.solver import Solver
import niftymic.base.stack as st


# This class implements the framework to iteratively solve
#  \f$ \vec{y}_k = A_k \vec{x} \f$ for every slice \f$ \vec{y}_k,\,k=1,\dots,K \f$
#  via Tikhonov-regularization via an augmented least-square approach
#  where \f$A_k=D_k B_k W_k\in\mathbb{R}^{N_k}\f$ denotes the combined warping, blurring and downsampling
#  operation, \f$ M_k \f$ the masking operator and \f$G\f$ represents either
#  the identity matrix \f$I\f$ (zeroth-order Tikhonov) or
#  the (flattened, stacked vector) gradient
#  \f$ \nabla  = \begin{pmatrix} D_x \\ D_y \\ D_z \end{pmatrix} \f$
#  (first-order Tikhonov).
#  The minimization problem reads
#  \f[
#       \text{arg min}_{\vec{x}} \Big( \sum_{k=1}^K \frac{1}{2} \Vert M_k (\vec{y}_k - A_k \vec{x} )\Vert_{\ell^2}^2
#                       + \frac{\alpha}{2}\,\Vert G\vec{x} \Vert_{\ell^2}^2 \Big)
#       =
#       \text{arg min}_{\vec{x}} \Bigg( \Bigg\Vert
#           \begin{pmatrix} M_1 A_1 \\ M_2 A_2 \\ \vdots \\ M_K A_K \\ \sqrt{\alpha} G \end{pmatrix} \vec{x}
#           - \begin{pmatrix} M_1 \vec{y}_1 \\ M_2 \vec{y}_2 \\ \vdots \\ M_K \vec{y}_K \\ \vec{0} \end{pmatrix}
#       \Bigg\Vert_{\ell^2}^2 \Bigg)
#  \f]
#  By defining the shorthand
#  \f[
#   MA := \begin{pmatrix} M_1 A_1 \\ M_2 A_2 \\ \vdots \\ M_K A_K \end{pmatrix}\in\mathbb{R}^{\sum_k N_k} \quad\text{and}\quad
#   M\vec{y} := \begin{pmatrix} M_1 \vec{y}_1 \\ M_2 \vec{y}_2 \\ \vdots \\ M_K \vec{y}_K \end{pmatrix}\in\mathbb{R}^{\sum_k N_k}
#  \f]
#  the problem can be compactly written as
#  \f[
#       \text{arg min}_{\vec{x}} \Bigg( \Bigg\Vert
#           \begin{pmatrix} MA \\ \sqrt{\alpha} G \end{pmatrix} \vec{x}
#           - \begin{pmatrix} M\vec{y} \\ \vec{0} \end{pmatrix}
#       \Bigg\Vert_{\ell^2}^2 \Bigg)
#  \f]
#  with \f$ G\in\mathbb{R}^N \f$ in case of \f$G=I\f$ or
#  \f$G\in\mathbb{R}^{3N}\f$ in case of \f$G\f$ representing the gradient.
#  \see \p itkAdjointOrientedGaussianInterpolateImageFilter of \p ITK
#  \see \p itOrientedGaussianInterpolateImageFunction of \p ITK
class TikhonovSolver(Solver):

    ##
    # Constructor
    # \date          2016-08-01 23:00:04+0100
    #
    # \param         self                   The object
    # \param         stacks                 list of Stack objects containing
    #                                       all stacks used for the
    #                                       reconstruction
    # \param[in,out] reconstruction         Stack object containing the current
    #                                       estimate of the reconstruction
    #                                       volume (used as initial value +
    #                                       space definition)
    # \param         alpha_cut              Cut-off distance for Gaussian
    #                                       blurring filter
    # \param         alpha                  regularization parameter, scalar
    # \param         iter_max               number of maximum iterations,
    #                                       scalar
    # \param         reg_type               Type of Tikhonov regualrization,
    #                                       i.e. TK0 or TK1 for either zeroth-
    #                                       or first order Tikhonov
    # \param         minimizer              Type of minimizer used to solve
    #                                       minimization problem, possible
    #                                       types: 'lsmr', 'lsqr', 'L-BFGS-B' #
    # \param         deconvolution_mode     Either "full_3D" or
    #                                       "only_in_plane". Indicates whether
    #                                       full 3D or only in-plane
    #                                       deconvolution is considered
    # \param         data_loss              The loss
    # \param         huber_gamma            The huber gamma
    # \param         predefined_covariance  The predefined covariance
    # \param         verbose                The verbose
    #
    def __init__(self,
                 stacks,
                 reconstruction,
                 alpha_cut=3,
                 alpha=0.03,
                 iter_max=10,
                 reg_type="TK1",
                 minimizer="lsmr",
                 deconvolution_mode="full_3D",
                 x_scale="max",
                 data_loss="linear",
                 data_loss_scale=1,
                 huber_gamma=1.345,
                 predefined_covariance=None,
                 use_masks=True,
                 verbose=1,
                 ):

        # Run constructor of superclass
        Solver.__init__(self,
                        stacks=stacks,
                        reconstruction=reconstruction,
                        alpha_cut=alpha_cut,
                        alpha=alpha,
                        iter_max=iter_max,
                        minimizer=minimizer,
                        deconvolution_mode=deconvolution_mode,
                        x_scale=x_scale,
                        data_loss=data_loss,
                        data_loss_scale=data_loss_scale,
                        huber_gamma=huber_gamma,
                        predefined_covariance=predefined_covariance,
                        verbose=verbose,
                        use_masks=use_masks,
                        )

        # Settings for optimizer
        self._reg_type = reg_type

    #
    # Set type of regularization. It can be either 'TK0' or 'TK1'
    # \date       2017-07-25 15:19:17+0100
    #
    # \param      self      The object
    # \param      reg_type  Either 'TK0' or 'TK1', string
    #
    # \return     { description_of_the_return_value }
    #
    def set_regularization_type(self, reg_type):
        self._reg_type = reg_type

    # Get chosen type of regularization.
    #  \return regularization type as string
    def get_regularization_type(self):
        return self._reg_type

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
    def get_setting_specific_filename(self, prefix="SRR_"):

        # Build filename
        filename = prefix
        filename += "stacks" + str(len(self._stacks))
        if self._alpha > 0:
            filename += "_" + self._reg_type
        filename += "_" + self._minimizer
        if self._data_loss not in ["linear"]:
            filename += "_" + self._data_loss
            if self._data_loss in ["huber"]:
                filename += str(self._huber_gamma)
            filename += "_fscale%g" % self._data_loss_scale
        filename += "_alpha" + str(self._alpha)
        filename += "_itermax" + str(self._iter_max)

        # Replace dots by 'p'
        filename = filename.replace(".", "p")

        return filename

    def get_solver(self):
        if self._reg_type not in ["TK0", "TK1"]:
            raise ValueError(
                "Error: regularization type can only be either 'TK0' or 'TK1'")

        # Get operators
        A = self.get_A()
        A_adj = self.get_A_adj()
        b = self.get_b()
        x0 = self.get_x0()
        x_scale = self.get_x_scale()

        if self._reg_type == "TK0":
            B = lambda x: x.flatten()
            B_adj = lambda x: x.flatten()

        elif self._reg_type == "TK1":
            spacing = np.array(self._reconstruction.sitk.GetSpacing())
            linear_operators = linop.LinearOperators3D(spacing=spacing)
            grad, grad_adj = linear_operators.get_gradient_operators()

            X_shape = self._reconstruction_shape
            Z_shape = grad(x0.reshape(*X_shape)).shape

            B = lambda x: grad(x.reshape(*X_shape)).flatten()
            B_adj = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

        # Set up solver
        solver = tk.TikhonovLinearSolver(
            A=A,
            A_adj=A_adj,
            B=B,
            B_adj=B_adj,
            b=b,
            x0=x0,
            x_scale=x_scale,
            alpha=self._alpha,
            data_loss=self._data_loss,
            data_loss_scale=self._data_loss_scale,
            verbose=self._verbose,
            minimizer=self._minimizer,
            iter_max=self._iter_max,
            bounds=(0, np.inf),
        )
        return solver

    ##
    # Run the reconstruction algorithm based on Tikhonov regularization
    # \date       2016-07-29 12:35:01+0100
    # \post       self._reconstruction is updated with new volume and can be
    #             fetched by \p get_recon
    #
    # \param      self  The object
    # \param      provide_initial_value  Use reconstruction volume during
    #                                    initialization as initial value, boolean.
    #                                    Otherwise, assume zero initial vale.
    #
    def _run(self):

        solver = self.get_solver()

        self._print_info_text()

        # Run reconstruction
        solver.run()

        # Get computational time
        self._computational_time = solver.get_computational_time()

        # After reconstruction: Update member attribute
        self._reconstruction.itk = self._get_itk_image_from_array_vec(
            solver.get_x(), self._reconstruction.itk)
        self._reconstruction.sitk = sitkh.get_sitk_from_itk_image(
            self._reconstruction.itk)

    def _print_info_text(self):

        ph.print_subtitle("Tikhonov Solver:")
        ph.print_info("Chosen regularization type: ", newline=False)
        if self._reg_type in ["TK0"]:
            print("Zeroth-order Tikhonov")

        else:
            print("First-order Tikhonov")

        if self._deconvolution_mode in ["only_in_plane"]:
            ph.print_info("(Only in-plane deconvolution is performed)")

        elif self._deconvolution_mode in ["predefined_covariance"]:
            ph.print_info("(Predefined covariance used: cov = %s)"
                          % (np.diag(self._predefined_covariance)))

        if self._data_loss in ["huber"]:
            ph.print_info("Loss function: %s (gamma = %g)" %
                          (self._data_loss, self._huber_gamma))
        else:
            ph.print_info("Loss function: %s" % (self._data_loss))

        if self._data_loss != "linear":
            ph.print_info("Loss function scale: %g" % (self._data_loss_scale))

        ph.print_info("Regularization parameter: " + str(self._alpha))
        ph.print_info("Minimizer: " + self._minimizer)
        ph.print_info(
            "Maximum number of iterations: " + str(self._iter_max))
        # ph.print_info("Tolerance: %.0e" %(self._tolerance))


class TemporalTikhonovSolver(object):

    def __init__(self,
                 stacks,
                 reconstruction,
                 alpha_cut=3,
                 alpha=0.03,
                 beta=0.1,
                 iter_max=10,
                 reg_type="TK1",
                 minimizer="lsmr",
                 deconvolution_mode="full_3D",
                 x_scale="max",
                 data_loss="linear",
                 data_loss_scale=1,
                 huber_gamma=1.345,
                 predefined_covariance=None,
                 use_masks=True,
                 verbose=1,
                 ):

        self._solvers = [
            TikhonovSolver(
                stacks=[s],
                reconstruction=st.Stack.from_stack(reconstruction),
                alpha_cut=alpha_cut,
                alpha=alpha,
                iter_max=iter_max,
                reg_type=reg_type,
                minimizer=minimizer,
                deconvolution_mode=deconvolution_mode,
                x_scale=x_scale,
                data_loss=data_loss,
                data_loss_scale=data_loss_scale,
                huber_gamma=huber_gamma,
                predefined_covariance=predefined_covariance,
                use_masks=use_masks,
                verbose=verbose,
            )
            for s in stacks
        ]

        self._reg_type = reg_type

        self._alpha = alpha
        self._beta = beta
        self._iter_max = iter_max
        self._verbose = verbose

        self._stacks = stacks
        self._reconstruction = reconstruction

        self._computational_time = None
        self._reconstructions = None
        self._bounds = (0, np.inf)

    def get_computational_time(self):
        return self._computational_time

    def get_reconstructions(self):
        return self._reconstructions

    def run(self):
        time_start = ph.start_timing()
        self._print_info_text()

        shape_x = self._solvers[0]._reconstruction_shape
        self._n_x = np.array(shape_x).prod()
        self._n_x_total = len(self._solvers) * self._n_x

        x0 = self._solvers[0].get_x0()
        if self._reg_type == "TK0":
            self._B = lambda x: x.flatten()
            self._B_adj = lambda x: x.flatten()

        elif self._reg_type == "TK1":
            spacing = np.array(self._reconstruction.sitk.GetSpacing())
            linear_operators = linop.LinearOperators3D(spacing=spacing)
            grad, grad_adj = linear_operators.get_gradient_operators()

            X_shape = shape_x
            Z_shape = grad(x0.reshape(*X_shape)).shape

            self._B = lambda x: grad(x.reshape(*X_shape)).flatten()
            self._B_adj = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

        self._B_shape = (self._B(x0).size, x0.size)

        n_rhs = 0
        self._rhs = []
        self._A = []
        self._A_adj = []
        self._D = []
        self._D_adj = []
        for solver in self._solvers:
            b = solver.get_b()
            n_rhs += len(b)
            self._rhs.append(b)
            self._A.append(solver.get_A())
            self._A_adj.append(solver.get_A_adj())

        if self._alpha > EPS:
            n_rhs += len(self._solvers) * self._B_shape[0]

        if self._beta > EPS:
            n_rhs += (len(self._solvers) - 1) * self._n_x

        self._x_1D = np.zeros(self._n_x_total)
        self._rhs_1D = np.zeros(n_rhs)

        A_fw = lambda x: self._A_fw(
            x, np.sqrt(self._alpha), np.sqrt(self._beta))
        A_bw = lambda x: self._A_bw(
            x, np.sqrt(self._alpha), np.sqrt(self._beta))

        # Construct (sparse) linear operator A
        A = scipy.sparse.linalg.LinearOperator(
            shape=(self._rhs_1D.size, self._x_1D.size),
            matvec=A_fw,
            rmatvec=A_bw)
        b = np.zeros_like(A(self._x_1D))
        b_upper = np.concatenate(self._rhs)
        b[:b_upper.size] = b_upper

        x = scipy.sparse.linalg.lsmr(
            A, b,
            maxiter=self._iter_max,
            show=self._verbose,
            atol=0,
            btol=0)[0]

        if self._bounds is not None:
            # Clip to bounds
            x = np.clip(x, self._bounds[0], self._bounds[1])

        self._reconstructions = self._get_reconstructions(x)

        # y_vec = s.get_b()

        self._computational_time = ph.stop_timing(time_start)

    def _A_fw(self, x, sqrt_alpha, sqrt_beta):

        i0 = 0

        # cost
        for i, solver in enumerate(self._solvers):
            i1 = i0 + len(self._rhs[i])
            self._rhs_1D[i0:i1] = self._A[i](
                x[i * self._n_x:(i + 1) * self._n_x]
            )
            i0 = i1

        # tikhonov
        if sqrt_alpha > EPS:
            for i, solver in enumerate(self._solvers):
                self._rhs_1D[
                    i0 + self._B_shape[0] * i:
                    i0 + self._B_shape[0] * (i + 1)
                ] = sqrt_alpha * self._B(x[i * self._n_x:(i + 1) * self._n_x])
                i1 = i0 + self._B_shape[0] * (i + 1)

        # temporal
        if sqrt_beta > EPS:
            for i in range(len(self._solvers) - 1):
                self._rhs_1D[
                    i1 + self._n_x * i:
                    i1 + self._n_x * (i + 1)
                ] = sqrt_beta * (
                    x[(i + 1) * self._n_x:(i + 2) * self._n_x]
                    - x[i * self._n_x:(i + 1) * self._n_x]
                )

        return self._rhs_1D

    def _A_bw(self, b, sqrt_alpha, sqrt_beta):
        i0 = 0
        self._x_1D[:] = 0

        # cost
        for i, solver in enumerate(self._solvers):
            i1 = i0 + len(self._rhs[i])
            self._x_1D[i * self._n_x:(i + 1) * self._n_x] = \
                self._A_adj[i](b[i0:i1])
            i0 = i1

        # tikhonov
        if sqrt_alpha > EPS:
            for i, solver in enumerate(self._solvers):
                self._x_1D[i * self._n_x:(i + 1) * self._n_x] += \
                    sqrt_alpha * self._B_adj(
                        b[i0:i0 + self._B_shape[0]]
                )
                i0 += self._B_shape[0]

        # temporal
        if sqrt_beta > EPS:
            i = 0
            self._x_1D[i * self._n_x:(i + 1) * self._n_x] += \
                - sqrt_beta * b[i0 + self._n_x * i: i0 + self._n_x * (i + 1)]

            for i in range(1, len(self._solvers) - 1):
                self._x_1D[i * self._n_x:(i + 1) * self._n_x] += \
                    sqrt_beta * (
                        b[i0 + self._n_x * (i - 1): i0 + self._n_x * i]
                        - b[i0 + self._n_x * i: i0 + self._n_x * (i + 1)]
                )

            i = len(self._solvers) - 1
            self._x_1D[i * self._n_x:(i + 1) * self._n_x] += \
                sqrt_beta * (
                    b[i0 + self._n_x * (i - 1): i0 + self._n_x * i]
            )

        return self._x_1D

    def _get_reconstructions(self, x):

        reconstructions = []
        for i, solver in enumerate(self._solvers):
            x_vec = x[i * self._n_x:(i + 1) * self._n_x]
            recon_itk = solver._get_itk_image_from_array_vec(
                x_vec, self._reconstruction.itk)
            recon_sitk = sitkh.get_sitk_from_itk_image(recon_itk)
            reconstructions.append(
                st.Stack.from_sitk_image(
                    image_sitk=recon_sitk,
                    slice_thickness=self._reconstruction.get_slice_thickness(),
                    image_sitk_mask=self._reconstruction.sitk_mask,
                )
            )
        return reconstructions

    def _print_info_text(self):

        ph.print_subtitle("Temporal Tikhonov Solver:")
        ph.print_info("Chosen regularization type: ", newline=False)
        if self._reg_type in ["TK0"]:
            print("Zeroth-order Tikhonov")

        else:
            print("First-order Tikhonov")

        # if self._deconvolution_mode in ["only_in_plane"]:
        #     ph.print_info("(Only in-plane deconvolution is performed)")

        # elif self._deconvolution_mode in ["predefined_covariance"]:
        #     ph.print_info("(Predefined covariance used: cov = %s)"
        #                   % (np.diag(self._predefined_covariance)))

        # if self._data_loss in ["huber"]:
        #     ph.print_info("Loss function: %s (gamma = %g)" %
        #                   (self._data_loss, self._huber_gamma))
        # else:
        #     ph.print_info("Loss function: %s" % (self._data_loss))

        # if self._data_loss != "linear":
        #     ph.print_info("Loss function scale: %g" % (self._data_loss_scale))

        ph.print_info(
            "Regularization parameter alpha (spatial reg): " + str(self._alpha))
        ph.print_info(
            "Regularization parameter beta (temporal reg): " + str(self._beta))
        # ph.print_info("Minimizer: " + self._minimizer)
        ph.print_info("Maximum number of iterations: " + str(self._iter_max))
        # ph.print_info("Tolerance: %.0e" %(self._tolerance))

    def get_setting_specific_filename(self, prefix="SRR_"):

        # Build filename
        filename = prefix
        filename += "stacks" + str(len(self._stacks))
        if self._alpha > 0 or self._beta > 0:
            filename += "_" + self._reg_type
        # filename += "_" + self._minimizer
        # if self._data_loss not in ["linear"]:
        #     filename += "_" + self._data_loss
        #     if self._data_loss in ["huber"]:
        #         filename += str(self._huber_gamma)
        #     filename += "_fscale%g" % self._data_loss_scale
        filename += "_alpha" + str(self._alpha)
        filename += "_beta" + str(self._beta)
        filename += "_itermax" + str(self._iter_max)

        # Replace dots by 'p'
        filename = filename.replace(".", "p")

        return filename
