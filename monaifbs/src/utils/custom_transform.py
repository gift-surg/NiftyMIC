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
# \file       custom_transform.py
# \brief      contains a series of custom dict transforms to be used in MONAI data preparation for the dynUnet model
#
# \author     Marta B M Ranzini (marta.ranzini@kcl.ac.uk)
# \date       November 2020

import numpy as np
import copy
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

from monai.config import KeysCollection
from monai.transforms import (
    DivisiblePad, MapTransform, Spacing, Spacingd
)
from monai.utils import (
    NumpyPadMode,
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)
NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]
GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]


class ConverToOneHotd(MapTransform):
    """
    Convert multi-class label to One Hot Encoding
    """

    def __init__(self, keys, labels):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            labels: list of labels to be converted to one-hot

        """
        super().__init__(keys)
        self.labels = labels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = list()
            for n in self.labels:
                result.append(d[key] == n)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class MinimumPadd(MapTransform):
    """
    Pad the input data, so that the spatial sizes are at least of size `k`.
    Dictionary-based wrapper of :py:class:`monai.transforms.DivisiblePad`.
    """

    def __init__(
        self, keys: KeysCollection, k: Union[Sequence[int], int], mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            k: the target k for each spatial dimension.
                if `k` is negative or 0, the original size is preserved.
                if `k` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
        See also :py:class:`monai.transforms.SpatialPad`
        """
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.k = k
        self.padder = DivisiblePad(k=k)

    def __call__(self, data):
        d = dict(data)
        for key, m in zip(self.keys, self.mode):
            spatial_shape = np.array(d[key].shape[1:])
            k = np.array(fall_back_tuple(self.k, (1,) * len(spatial_shape)))
            if np.any(spatial_shape < k):
                d[key] = self.padder(d[key], mode=m)
        return d


class InPlaneSpacingd(Spacingd):
    """
    Performs the same operation as the MONAI Spacingd transform, but allows to preserve the spacing along some axes,
    which should be indicated as -1.0 in the input pixdim.
    E.g. pixdim=(0.8, 0.8, -1.0) would change the x-y plane spacing to (0.8, 0.8) while preserving the original
    spacing along z.
    See also :py:class: `monai.transforms.Spacingd`
    """
    def __init__(self,
                 keys: KeysCollection,
                 pixdim: Sequence[float],
                 diagonal: bool = False,
                 mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
                 padding_mode: GridSamplePadModeSequence = GridSamplePadMode.BORDER,
                 align_corners: Union[Sequence[bool], bool] = False,
                 dtype: Optional[Union[Sequence[np.dtype], np.dtype]] = np.float64,
                 meta_key_postfix: str = "meta_dict",
                 ) -> None:
        """
        Args
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            pixdim: output voxel spacing.
            diagonal: whether to resample the input to have a diagonal affine matrix.
            mode: {``"bilinear"``, ``"nearest"``}. Interpolation mode to calculate output values.
                Defaults to ``"bilinear"``.
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            align_corners:  Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
                default is `meta_dict`, the meta data is a dictionary object.
        See also :py:class: `monai.transforms.Spacingd` for more information on the inputs
        """
        super().__init__(keys,
                         pixdim,
                         diagonal,
                         mode,
                         padding_mode,
                         align_corners,
                         dtype,
                         meta_key_postfix)
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.dim_to_keep = np.argwhere(self.pixdim == -1.0)

    def __call__(self,
                 data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
                 ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            meta_data = d[f"{key}_{self.meta_key_postfix}"]
            # set pixdim to original pixdim value where required
            current_pixdim = copy.deepcopy(self.pixdim)
            original_pixdim = meta_data["pixdim"]
            old_pixdim = original_pixdim[1:4]
            current_pixdim[self.dim_to_keep] = old_pixdim[self.dim_to_keep]

            # apply the transform
            spacing_transform = Spacing(current_pixdim, diagonal=self.diagonal)

            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            d[key], _, new_affine = spacing_transform(
                data_array=d[key],
                affine=meta_data["affine"],
                mode=self.mode[idx],
                padding_mode=self.padding_mode[idx],
                align_corners=self.align_corners[idx],
                dtype=self.dtype[idx],
            )

            # store the modified affine
            meta_data["affine"] = new_affine
        return d
