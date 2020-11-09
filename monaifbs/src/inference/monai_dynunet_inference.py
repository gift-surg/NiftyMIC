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

import os
import sys
import yaml
import argparse
import logging
import torch

from torch.utils.data import DataLoader

from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.networks.nets import DynUNet
from monai.engines import SupervisedEvaluator
from monai.handlers import CheckpointLoader, SegmentationSaver, StatsHandler
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    NormalizeIntensityd,
    ToTensord,
    Activationsd,
    AsDiscreted,
    KeepLargestConnectedComponentd
)

from monaifbs.src.utils.custom_inferer import SlidingWindowInferer2D
from monaifbs.src.utils.custom_transform import InPlaneSpacingd


def create_data_list_of_dictionaries(input_files):

    print("*** Input data: ")
    full_list = []
    # convert to list if single file
    if type(input_files) is str:
        input_files = [input_files]
    for current_f in input_files:
        if os.path.isfile(current_f):
                print(current_f)
                full_list.append({"image": current_f})
        else:
            raise FileNotFoundError('Expected image file: {} not found'.format(current_f))
    return full_list


def run_inference(input_data, config_info):
    """
    Read input and configuration parameters
    """

    val_files = create_data_list_of_dictionaries(input_data)

    # print MONAI config information
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print("*** MONAI config: ")
    print_config()

    # print to log the parameter setups
    print("*** Network inference config: ")
    print(yaml.dump(config_info))

    # inference params
    nr_out_channels = config_info['inference']['nr_out_channels']
    spacing = config_info["inference"]["spacing"]
    prob_thr = config_info['inference']['probability_threshold']
    model_to_load = config_info['inference']['model_to_load']
    if not os.path.exists(model_to_load):
        raise FileNotFoundError('Trained model not found')
    patch_size = config_info["inference"]["inplane_size"] + [1]
    print("Considering patch size = {}".format(patch_size))

    # set up either GPU or CPU usage
    if torch.cuda.is_available():
        print("\n#### GPU INFORMATION ###")
        print("Using device number: {}, name: {}".format(torch.cuda.current_device(), torch.cuda.get_device_name()))
        current_device = torch.device("cuda:0")
    else:
        current_device = torch.device("cpu")
        print("Using device: {}".format(current_device))

    """
    Data Preparation
    """
    print("***  Preparing data ... ")
    # data preprocessing for inference:
    # - convert data to right format [batch, channel, dim, dim, dim]
    # - resample to the training resolution in-plane (not along z)
    # - apply whitening
    # - convert to tensor
    val_transforms = Compose(
        [
            LoadNiftid(keys=["image"]),
            AddChanneld(keys=["image"]),
            InPlaneSpacingd(
                keys=["image"],
                pixdim=spacing,
                mode="bilinear",
            ),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )
    # create a validation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            num_workers=config_info['device']['num_workers'])

    def prepare_batch(batchdata):
        assert isinstance(batchdata, dict), "prepare_batch expects dictionary input data."
        return (
            (batchdata["image"], batchdata["label"])
            if "label" in batchdata
            else (batchdata["image"], None)
        )

    """
    Network preparation
    """
    print("***  Preparing network ... ")
    spacings = spacing[:2]
    sizes = patch_size[:2]
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    print("strides:")
    print(strides)
    print("kernels:")
    print(kernels)

    net = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=nr_out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=2,
        res_block=False
    ).to(current_device)

    """
    Set ignite evaluator to perform inference
    """
    print("***  Preparing evaluator ... ")
    if nr_out_channels == 1:
        do_sigmoid = True
        do_softmax = False
    elif nr_out_channels > 1:
        do_sigmoid = False
        do_softmax = True
    else:
        raise Exception("incompatible number of output channels")
    print("Using sigmoid={} and softmax={} as final activation".format(do_sigmoid, do_softmax))
    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=do_sigmoid, softmax=do_softmax),
            AsDiscreted(keys="pred", argmax=True, threshold_values=True, logit_thresh=prob_thr),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=1)
        ]
    )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointLoader(load_path=model_to_load, load_dict={"net": net}, map_location=torch.device('cpu')),
        SegmentationSaver(
            output_dir=config_info['output']['out_dir'],
            output_ext='.nii.gz',
            output_postfix=config_info['output']['out_postfix'],
            batch_transform=lambda batch: batch["image_meta_dict"],
            output_transform=lambda output: output["pred"],
        ),
    ]

    # Define customized evaluator
    class DynUNetEvaluator(SupervisedEvaluator):
        def _iteration(self, engine, batchdata):
            inputs, targets = self.prepare_batch(batchdata)
            inputs = inputs.to(engine.state.device)
            if targets is not None:
                targets = targets.to(engine.state.device)
            flip_inputs_1 = torch.flip(inputs, dims=(2,))
            flip_inputs_2 = torch.flip(inputs, dims=(3,))
            flip_inputs_3 = torch.flip(inputs, dims=(2, 3))

            def _compute_pred():
                pred = self.inferer(inputs, self.network)
                flip_pred_1 = torch.flip(self.inferer(flip_inputs_1, self.network), dims=(2,))
                flip_pred_2 = torch.flip(self.inferer(flip_inputs_2, self.network), dims=(3,))
                flip_pred_3 = torch.flip(self.inferer(flip_inputs_3, self.network), dims=(2, 3))
                return (pred + flip_pred_1 + flip_pred_2 + flip_pred_3) / 4

            # execute forward computation
            self.network.eval()
            with torch.no_grad():
                if self.amp:
                    with torch.cuda.amp.autocast():
                        predictions = _compute_pred()
                else:
                    predictions = _compute_pred()
            return {"image": inputs, "label": targets, "pred": predictions}

    evaluator = DynUNetEvaluator(
        device=current_device,
        val_data_loader=val_loader,
        network=net,
        prepare_batch=prepare_batch,
        inferer=SlidingWindowInferer2D(roi_size=patch_size, sw_batch_size=4, overlap=0.0),
        post_transform=val_post_transforms,
        val_handlers=val_handlers,
        amp=False,
    )

    """
    Run inference
    """
    print("***  Running evaluator ... ")
    evaluator.run()
    print("Done!")

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run inference with dynUnet with MONAI.')
    parser.add_argument('--config',
                        dest='config',
                        metavar='config',
                        type=str,
                        help='config file containing network information for inference',
                        required=True)
    parser.add_argument('--in_files',
                        dest='in_files',
                        metavar='in_files',
                        type=str,
                        nargs='+',
                        help='all files to be processed',
                        required=True)
    parser.add_argument('--out_folder',
                        dest='out_folder',
                        metavar='out_folder',
                        type=str,
                        help='directory where to store the outputs',
                        required=True)
    parser.add_argument('--out_postfix',
                        dest='out_postfix',
                        metavar='out_postfix',
                        type=str,
                        help='postfix to add to the input names for the output filename',
                        default='_seg')
    args = parser.parse_args()

    # read the config file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # read the input files
    in_files = args.in_files

    # add the output directory to the config dictionary
    config['output']['out_dir'] = args.out_folder
    if not os.path.exists(config['output']['out_dir']):
        os.makedirs(config['output']['out_dir'])
    config['output']['out_postfix'] = args.out_postfix

    # run inference with MONAI dynUnet
    run_inference(in_files, config)
