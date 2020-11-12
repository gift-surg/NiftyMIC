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
# \file       custom_losses.py
# \brief      contains a series of loss functions that can be used to train the dynUNet model
#               The code is inspired and includes some modifications to the
#               DiceLoss implementation in MONAI
#               https://github.com/Project-MONAI/MONAI/blob/releases/0.3.0/monai/losses/dice.py
#               and the losses included in the dynUNet tutorial in MONAI
#               https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_tutorial.ipynb
#
# \author     Marta B M Ranzini (marta.ranzini@kcl.ac.uk)
# \date       November 2020

import warnings
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from monai.networks.utils import one_hot
from monai.utils import LossReduction


class DiceLossExtended(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth` parameter is a value added to the
    intersection and union components of the inter-over-union calculation to smooth results and prevent divide by 0,
    this value should be small. The `include_background` class attribute can be set to False for an instance of
    DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be background.
    If the non-background segmentations are small compared to the total image size they can get overwhelmed by
    the signal from the background so excluding it in such cases helps convergence.
    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation,
    3DV, 2016.

    With respect to monai.losses.DiceLoss, this implementation allows for:
    - the use of a "Batch Dice" (batch version) as in the nnUNet implementation. The Dice is computed for the whole
        batch (1 value per class channel), as opposed to being computed for each element in the batch and then averaged
        across the batch.
    - the selection of different smooth terms at numerator and denominator.
    - the possibility to define a power term (pow) for the Dice, such as the returned loss is Dice^pow
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        pow: float = 1.,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        batch_version: bool = False,
        smooth_num: float = 1e-5,
        smooth_den: float = 1e-5
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            pow: raise the Dice to the required power (default 1)
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch_version: if True, a single Dice value is computed for the whole batch per class. If False, the Dice
                is computed per element in the batch and then reduced (sum/average/None) across the batch.
            smooth_num: a small constant to be added to the numerator of Dice to avoid nan.
            smooth_den: a small constant to be added to the denominator of Dice to avoid nan.
        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.pow = pow
        self.jaccard = jaccard
        self.batch_version = batch_version
        self.smooth_num = smooth_num
        self.smooth_den = smooth_den

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD]
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        if self.batch_version:
            # reducing only spatial dimensions and batch (not channels)
            reduce_axis = [0] + list(range(2, len(input.shape)))
        else:
            # reducing only spatial dimensions (not batch nor channels)
            reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = (1.0 - (2.0 * intersection + self.smooth_num) / (denominator + self.smooth_den)) ** self.pow

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses or [n_classes] if batch version
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


# CUSTOM LOSSES FROM dynUNet tutorial for dynUNet training
# code from https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_tutorial.ipynb
class CrossEntropyLoss(nn.Module):
    """
    Compute the multi-channel cross entropy between predictions and ground truth.
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: the shape should be BNH[WD].
            y_true: the shape should be BNH[WD]
        """
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        y_true = torch.squeeze(y_true, dim=1).long()
        return self.loss(y_pred, y_true)


class DiceCELoss(nn.Module):
    """
    Compute the loss function = Dice + Cross Entropy.
    The monaifbs.src.utils.custom_losses.DiceLossExtended class is used to compute the Dice score, which gives
    flexibility on the type of Dice to compute (e.g. use the Dice per image averaged across the batch
    or the Batch Dice).
    The monaifbs.src.utils.custom_losses.CrossEntropyLoss class is used to compute the cross entropy.
    """
    def __init__(self,
                 include_background: bool = True,
                 to_onehot_y: bool = True,
                 sigmoid: bool = False,
                 softmax: bool = True,
                 other_act: Optional[Callable] = None,
                 squared_pred: bool = False,
                 pow: float = 1.,
                 jaccard: bool = False,
                 reduction: Union[LossReduction, str] = LossReduction.MEAN,
                 batch_version: bool = False,
                 smooth_num: float = 1e-5,
                 smooth_den: float = 1e-5
                 ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            pow: raise the Dice to the required power (default 1)
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch_version: if True, a single Dice value is computed for the whole batch per class. If False, the Dice
                is computed per element in the batch and then reduced (sum/average/None) across the batch.
            smooth_num: a small constant to be added to the numerator of Dice to avoid nan.
            smooth_den: a small constant to be added to the denominator of Dice to avoid nan.
        """
        super().__init__()
        self.dice = DiceLossExtended(include_background=include_background,
                                     to_onehot_y=to_onehot_y,
                                     sigmoid=sigmoid,
                                     softmax=softmax,
                                     other_act=other_act,
                                     squared_pred=squared_pred,
                                     pow=pow,
                                     jaccard=jaccard,
                                     reduction=reduction,
                                     batch_version=batch_version,
                                     smooth_num=smooth_num,
                                     smooth_den=smooth_den)
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        """
        Args
            y_pred: the shape should be BNH[WD].
            y_true: the shape should be BNH[WD].
        """
        dice = self.dice(y_pred, y_true)
        cross_entropy = self.cross_entropy(y_pred, y_true)
        return dice + cross_entropy
