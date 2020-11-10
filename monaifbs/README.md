# Fetal brain segmentation with MONAI DynUNet

monaifbs (MONAI Fetal Brain Segmentation) is a Pytorch-based toolkit to train and test deep learning models for automated 
fetal brain segmentation in HASTE-like MR images.
The toolkit was developed within the [GIFT-Surg][giftsurg] research project, and takes advantage of [MONAI][monai], 
a freely available, community-supported, PyTorch-based framework for deep learning in healthcare imaging.

A pre-trained dynUNet model is [provided][dynUnetmodel] and can be directly used for inference on new data using 
the script `src/inference/monai_dynunet_inference.py`. Alternatively, the script `fetal_brain_seg.py` provides
the same inference functionality within an appropriate interface to be used within the niftymic package and by the 
executable command `niftymic_segment_fetal_brains`. See the sections [Inference][inference_section] and 
[Use within NiftyMIC][use_section] below.

More information about MONAI dynUNet can be found [here][dynUnettutorial]. This deep learning pipeline is based on the 
[nnU-Net][nnunet] self-adapting framework for U-Net-based medical image segmentation. 

### Contact information
This package was developed by [Marta B.M. Ranzini][mranzini] at the [Department of Surgical and Interventional Sciences][sie], 
[King's College London (KCL)][kcl] (2020).
If you have any questions or comments, please drop an email to `marta.ranzini@kcl.ac.uk`.

## Installation
Please follow the [installation instructions][installation] for NiftyMIC. To use monaifbs, please make sure you install
all Python and Pytorch dependencies by running the following three commands sequentially:  
`pip install -r requirements.txt`  
`pip install -r requirements-monaifbs.txt`  
`pip install -e .`

*Note*: MONAI and monaifbs require Python versions >= 3.6.


## Training
TODO

## Inference
Inference can be run with the provided inference script with the following command:  
```
python <path_to_monaifbs>/src/inference/monai_dynunet_inference.py \
--in_files <path-to-img1.nii.gz> <path-to-img2.nii.gz> ... <path-to-imgN.nii.gz> \ 
--out_folder <path-to-output-directory> 
```

By default, this will use the provided [pre-trained model][dynUnetmodel] and the network configuration parameters
reported in this [config file][inference_config]. If you want to specify a different (MONAI dynUNet) trained model,
you can create your own config file indicating the model to load and its network configuration parameters
following the provided [template][inference_config]. Then, you can simply run inference as:  
```
python <path_to_monaifbs>/src/inference/monai_dynunet_inference.py \
--in_files <path-to-img1.nii.gz> <path-to-img2.nii.gz> ... <path-to-imgN.nii.gz> \ 
--out_folder <path-to-output-directory> \  
--config_file <path-to-custom-config.yml>
```

## Use within NiftyMIC (for inference)
By default, NiftyMIC uses monaifbs utilities to automatically generate fetal brain segmentation masks that can be used
for the reconstruction pipeline.
Provided the dependencies for monaifbs are installed, create the automatic fetal brain masks of HASTE-like images with 
the command:
```
niftymic_segment_fetal_brains \
--filenames \
nifti/name-of-stack-1.nii.gz \
nifti/name-of-stack-2.nii.gz \
nifti/name-of-stack-N.nii.gz \
--filenames-masks \
seg/name-of-stack-1.nii.gz \
seg/name-of-stack-2.nii.gz \
seg/name-of-stack-N.nii.gz
```

The interface between the niftymic package and monaifbs inference utilities is defined in `fetal_brain_seg.py`.
Its working scheme is essentially the same as the `monai_dynunet_inference.py`, it simply provides a wrapper to feed
the input data to the `run_inference()` function in `monai_dynunet_inference.py`.
It can also be used as a standalone script for inference as follows:  
```
python <path_to_monaifbs>/fetal_brain_seg.py \
--input_names <path-to-img1.nii.gz> <path-to-img2.nii.gz> ... <path-to-imgN.nii.gz> \ 
--segment_output_names <path-to-seg1.nii.gz> <path-to-seg2.nii.gz> ... <path-to-segN.nii.gz> 
```

## Troubleshooting
#### Issue with ParallelNative on MacOS

A warning message from ParallelNative is shown and the computation gets stuck. This issue appears to happen only on 
MacOS and is known to be linked to PyTorch DataLoader (as reported in https://github.com/pytorch/pytorch/issues/46409)

Warning message: `[W ParallelNative.cpp:206] Warning: Cannot set number of intraop threads after parallel work has started 
or after set_num_threads call when using native parallel backend (function set_num_threads` 

When observed: MacOS, Python 3.6 and Python 3.7, running on CPU.

Solution: add `OMP_NUM_THREADS=1` before the call of monaifbs scripts.  
Example 1:
```
OMP_NUM_THREADS=1 python <path_to_monaifbs>/src/inference/monai_dynunet_inference.py \
--in_files <path-to-img1.nii.gz> <path-to-img2.nii.gz> ... <path-to-imgN.nii.gz> \ 
--out_folder <path-to-output-directory> 
```
Example 2:
```
OMP_NUM_THREADS=1 niftymic_segment_fetal_brains \
--filenames \
nifti/name-of-stack-1.nii.gz \
nifti/name-of-stack-2.nii.gz \
nifti/name-of-stack-N.nii.gz \
--filenames-masks \
seg/name-of-stack-1.nii.gz \
seg/name-of-stack-2.nii.gz \
seg/name-of-stack-N.nii.gz
```



## Licensing and Copyright
Copyright (c) 2020 Marta Bianca Maria Ranzini and contributors.
This framework is made available as free open-source software under the [BSD-3-Clause License][bsd]. 
Other licenses may apply for dependencies.

[giftsurg]: http://www.gift-surg.ac.uk
[kcl]: https://www.kcl.ac.uk
[sie]: https://www.kcl.ac.uk/bmeis/our-departments/surgical-interventional-engineering
[monai]: https://monai.io/
[installation]: https://github.com/gift-surg/NiftyMIC/wiki/niftymic-installation
[dynUnetmodel]: TODO
[inference_section]: TODO
[use_section]: TODO
[dynUnettutorial]: https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_tutorial.ipynb
[nnunet]: https://arxiv.org/abs/1809.10486
[inference_config]: TODO
[mranzini]: www.linkedin.com/in/marta-bianca-maria-ranzini
[bsd]: https://opensource.org/licenses/BSD-3-Clause