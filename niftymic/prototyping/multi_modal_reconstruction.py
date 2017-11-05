
import itk
import numpy as np
import SimpleITK as sitk

import pysitk.simple_itk_helper as sitkh

import niftymic.reconstruction.linear_operators as lin_op
import scipy


class MRILaw(object):

    @staticmethod
    def f(S0, TR, TE, T1, T2):
        return S0 * (1. - np.exp(-TR/T1)) * np.exp(-TE/T2)

    @staticmethod
    def df_dS0_pointwise(S0, TR, TE, T1, T2):
        return (1. - np.exp(-TR/T1)) * np.exp(-TE/T2)

    @staticmethod
    def df_dT1_pointwise(S0, TR, TE, T1, T2):
        return S0 / np.exp(TR/T1 + TE/T2) * TR/(T1 * T1)

    @staticmethod
    def df_dT2_pointwise(S0, TR, TE, T1, T2):
        return S0 * (np.exp(-TR/T1) - 1.) * np.exp(-TE/T2) * TE/(T2 * T2)


class MultiModalReconstruction(object):

    def __init__(self,
                 stacks_dic,
                 TE_dic,
                 TR_dic,
                 reconstruction,
                 image_type=itk.Image.D3):

        self._stacks_dic = stacks_dic
        self._TE_dic = TE_dic
        self._TR_dic = TR_dic
        self._reconstruction = reconstruction

        self._N_modalities = len(self._stacks_dic.keys())
        self._N_voxels = np.prod(self._reconstruction.sitk.GetSize())

        # Create PyBuffer object for conversion between NumPy arrays and ITK
        # images
        self._itk2np = itk.PyBuffer[image_type]

        self._linear_operators = lin_op.LinearOperators()

        self._T1 = None
        self._T2 = None
        self._S0 = None

    def run(self):

        x0 = np.zeros(self._N_voxels * (2 + self._N_modalities))
        bounds_T1 = [[0, 1.]]
        bounds_T2 = [[0, 1.]]
        bounds = [[0, np.inf]] * x0.size

        bounds[-2*self._N_voxels:-self._N_voxels] = bounds_T1 * self._N_voxels
        bounds[-self._N_voxels:] = bounds_T2 * self._N_voxels

        S00 = sitk.GetArrayFromImage(self._reconstruction.sitk).flatten()
        T10 = 0.4
        T20 = 0.04
        x0[0:self._N_modalities * self._N_voxels] = np.repeat(
            S00, self._N_modalities)
        x0[-2*self._N_voxels:-self._N_voxels] = T10
        x0[-self._N_voxels:] = T20

        cost = lambda x: self._cost(x)
        jac = lambda x: self._grad_cost(x)

        recon = scipy.optimize.minimize(
            method="L-BFGS-B",
            fun=cost,
            jac=jac,
            x0=x0,
            bounds=bounds,
            options={'maxiter': 10, 'disp': True}).x

        S0 = [recon[m*self._N_voxels:(m+1)*self._N_voxels]
              for m in range(self._N_modalities)]
        T1 = recon[-2*self._N_voxels:-self._N_voxels]
        T2 = recon[-self._N_voxels:]

        S0_sitk = [self._get_sitk_image_from_array(
            S0[m], self._reconstruction.sitk)
            for m in range(self._N_modalities)]
        T1_sitk = self._get_sitk_image_from_array(
            T1, self._reconstruction.sitk)
        T2_sitk = self._get_sitk_image_from_array(
            T2, self._reconstruction.sitk)

        foo = []
        label = []
        # foo.extend(S0_sitk)
        foo.append(T1_sitk)
        label.append("T1")
        foo.append(T2_sitk)
        label.append("T2")
        sitkh.show_sitk_image(foo, label=label)

    def _cost(self, x):

        S0 = x[0:self._N_modalities*self._N_voxels]
        T1 = x[-2*self._N_voxels:-self._N_voxels]
        T2 = x[-self._N_voxels:]
        cost = 0
        for m in range(self._N_modalities):
            S0_m = S0[m*self._N_voxels:(m+1)*self._N_voxels]
            TE_m = self._TE_dic[m]
            TR_m = self._TR_dic[m]
            stacks = self._stacks_dic[m]
            for stack in stacks:
                for slice in stack.get_slices():
                    residual = self._residual_slice(
                        slice, S0_m, TR_m, TE_m, T1, T2)
                    cost += np.sum(np.square(residual))
        return cost * 0.5

    def _grad_cost(self, x):
        S0 = x[0:self._N_modalities*self._N_voxels]
        T1 = x[-2*self._N_voxels:-self._N_voxels]
        T2 = x[-self._N_voxels:]

        grad_S0 = np.zeros(self._N_voxels * self._N_modalities)
        grad_T1 = np.zeros(self._N_voxels)
        grad_T2 = np.zeros(self._N_voxels)

        for m in range(self._N_modalities):
            S0_m = S0[m*self._N_voxels:(m+1)*self._N_voxels]
            TE_m = self._TE_dic[m]
            TR_m = self._TR_dic[m]
            stacks = self._stacks_dic[m]
            for stack in stacks:
                for slice in stack.get_slices():
                    residual = self._residual_slice(
                        slice, S0_m, TR_m, TE_m, T1, T2)
                    residual_itk = self._get_itk_image_from_array(
                        residual, slice.itk)
                    A_k_adj_residual_itk = self._linear_operators.A_adj_itk(
                        residual_itk, self._reconstruction.itk)
                    A_k_adj_residual = self._itk2np.GetArrayFromImage(
                        A_k_adj_residual_itk).flatten()

                    # grad_S0:
                    b = MRILaw.df_dS0_pointwise(S0_m, TR_m, TE_m, T1, T2)
                    grad_S0[m*self._N_voxels:(m+1)*self._N_voxels] += \
                        b.flatten() * A_k_adj_residual

                    # grad_T1
                    b = MRILaw.df_dT1_pointwise(S0_m, TR_m, TE_m, T1, T2)
                    grad_T1 += b.flatten() * A_k_adj_residual

                    # grad_T2
                    b = MRILaw.df_dT2_pointwise(S0_m, TR_m, TE_m, T1, T2)
                    grad_T2 += b.flatten() * A_k_adj_residual

        return np.concatenate((grad_S0, grad_T1, grad_T2))

    def _residual_slice(self, slice, S0, TR, TE, T1, T2):

        y_k = self._y_k(slice)
        x_m_itk = self._x_m_itk(S0, TR, TE, T1, T2)
        A_k_x_m_itk = self._linear_operators.A_itk(x_m_itk, slice.itk)
        A_k_x_m = self._itk2np.GetArrayFromImage(A_k_x_m_itk).flatten()

        return A_k_x_m - y_k

    @staticmethod
    def _y_k(slice):
        slice_sitk = slice.sitk * \
            sitk.Cast(slice.sitk_mask, slice.sitk.GetPixelIDValue())
        return sitk.GetArrayFromImage(slice_sitk).flatten()

    def _x_m_itk(self, S0, TR, TE, T1, T2):
        x = MRILaw.f(S0, TR, TE, T1, T2)
        x_itk = self._get_itk_image_from_array(x, self._reconstruction.itk)
        return x_itk

    def _get_itk_image_from_array(self, nda, image_itk_ref):

        shape_nda = np.array(
            image_itk_ref.GetLargestPossibleRegion().GetSize())[::-1]

        image_itk = self._itk2np.GetImageFromArray(nda.reshape(shape_nda))
        image_itk.SetOrigin(image_itk_ref.GetOrigin())
        image_itk.SetSpacing(image_itk_ref.GetSpacing())
        image_itk.SetDirection(image_itk_ref.GetDirection())

        return image_itk

    def _get_sitk_image_from_array(self, nda, image_sitk_ref):
        shape_nda = np.array(image_sitk_ref.GetSize())[::-1]
        image_sitk = sitk.GetImageFromArray(nda.reshape(shape_nda))
        image_sitk.CopyInformation(image_sitk_ref)
        return image_sitk
