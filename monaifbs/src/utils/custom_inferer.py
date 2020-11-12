# Copyright 2020 Marta Bianca Maria Ranzini and contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##
# \file       custom_inferer.py
# \brief      contains a series of classes to adapt the MONAI SlidingWindowInferer to the case of feeding slices
#               from a 3D volume into a 2D network.
#               Adapted from the MONAI class SlidingWindowInferer
#               https://github.com/Project-MONAI/MONAI/blob/releases/0.3.0/monai/inferers/inferer.py
#
# \author     Marta B M Ranzini (marta.ranzini@kcl.ac.uk)
# \date       November 2020

import copy
import torch
from typing import Union

from monai.inferers.utils import sliding_window_inference
from monai.inferers import Inferer
from monai.utils import BlendMode


class Predict2DFrom3D:
    """
    Crop 2D slices from 3D inputs and perform 2D predictions.
    Args:
        predictor (Network): trained network to perform the prediction
    """
    def __init__(self,
                 predictor):
        self.predictor = predictor

    def __call__(self, data):
        """
        Callable function to perform the prediction on input data given the defined predictor (network) after
        squeezing dimensions = 1. The removed dimension is added back after the prediction.
        Args:
            data: torch.tensor, model input data for inference.
        :return:
        """
        # squeeze dimensions equal to 1
        orig_size = list(data.shape)
        data_size = list(data.shape[2:])
        for idx_dim in range(2, 2+len(data_size)):
            if data_size[idx_dim-2] == 1:
                data = torch.squeeze(data, dim=idx_dim)
        predictions = self.predictor(data)  # batched patch segmentation
        new_size = copy.deepcopy(orig_size)
        new_size[1] = predictions.shape[1]   # keep original data shape, but take channel dimension from the prediction
        predictions = torch.reshape(predictions, new_size)
        return predictions


class SlidingWindowInferer2D(Inferer):
    """
    Sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().
    Modified from monai.inferers.SlidingWindowInferer to squeeze the extra dimension derived from cropping slices from a
    3D volume. In other words, reduces the input from [B, C, H, W, 1] to [B, C, H, W] for the forward pass through the
    network and then reshapes it back to [B, C, H, W, 1], before stitching all the patches back together.

    Args:
        roi_size (list, tuple): the window size to execute SlidingWindow evaluation.
            If it has non-positive components, the corresponding `inputs` size will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

    Note:
        the "sw_batch_size" here is to run a batch of window slices of 1 input image,
        not batch size of input images.

    """

    def __init__(
        self, roi_size, sw_batch_size: int = 1, overlap: float = 0.25, mode: Union[BlendMode, str] = BlendMode.CONSTANT
    ):
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)

    def __call__(self, inputs: torch.Tensor, network):
        """
        Unified callable function API of Inferers.

        Args:
            inputs (torch.tensor): model input data for inference.
            network (Network): target model to execute inference.

        """
        # convert the network to a callable that squeezes 3D slices to 2D before performing the network prediction
        predictor_2d = Predict2DFrom3D(network)
        return sliding_window_inference(inputs, self.roi_size, self.sw_batch_size,
                                        predictor_2d, self.overlap, self.mode)


class SlidingWindowInferer2DWithResize(Inferer):
    """
    Sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().
    At inference, it applies a "resize" operation for the first two dimensions to match the network input size.
    After the forward pass, the network output is resized back to the original size.

    Args:
        roi_size (list, tuple): the window size to execute SlidingWindow evaluation.
            If it has non-positive components, the corresponding `inputs` size will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

    Note:
        the "sw_batch_size" here is to run a batch of window slices of 1 input image,
        not batch size of input images.

    """

    def __init__(
        self, roi_size, sw_batch_size: int = 1, overlap: float = 0.25, mode: Union[BlendMode, str] = BlendMode.CONSTANT
    ):
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)

    def __call__(self, inputs: torch.Tensor, network):
        """
        Unified callable function API of Inferers.

        Args:
            inputs (torch.tensor): model input data for inference.
            network (Network): target model to execute inference.

        """
        # resize the input to the appropriate network input
        orig_size = list(inputs.shape)
        resized_size = copy.deepcopy(orig_size)
        resized_size[2] = self.roi_size[0]
        resized_size[3] = self.roi_size[1]
        inputs_resize = torch.nn.functional.interpolate(inputs, size=resized_size[2:], mode='trilinear')

        # convert the network to a callable that squeezes 3D slices to 2D before performing the network prediction
        predictor_2d = Predict2DFrom3D(network)
        outputs = sliding_window_inference(inputs_resize, self.roi_size, self.sw_batch_size,
                                           predictor_2d, self.overlap, self.mode)

        # resize back to original size
        outputs = torch.nn.functional.interpolate(outputs, size=orig_size[2:], mode='nearest')
        return outputs