# Volumetric MRI Reconstruction from Motion Corrupted 2D Slices

This is a research-focused toolkit developed within the [GIFT-Surg][giftsurg] project  to reconstruct an isotropic, high-resolution volume from multiple, possibly motion-corrupted, stacks of low-resolution 2D slices. The framework relies on slice-to-volume registration algorithms for motion correction and reconstruction-based Super-Resolution (SR) techniques for the volumetric reconstruction. 
The entire reconstruction pipeline is programmed in Python by using a mix of SimpleITK, WrapITK and standard C++ITK.

The algorithm and software were developed by [Michael Ebner][mebner] at the [Translational Imaging Group][tig] in the [Centre for Medical Image Computing][cmic] at [University College London (UCL)][ucl].

If you have any questions or comments (or find bugs), please drop an email to `michael.ebner.14@ucl.ac.uk`.

# Features

Several methods have been implemented to solve the **Super-Resolution Reconstruction** (SRR) problem
```math
\vec{x}^* := \text{argmin}_{\vec{x}} \Big[\sum_{k=1}^K \sum_{i=1}^{N_k} \varrho\big( (A_k\vec{x} - \vec{y}_k)_i^2 \big) + \alpha\,\text{Reg}(\vec{x}) \Big]
```

to obtain the (vectorized) high-resolution 3D volume $`\vec{x}\in\mathbb{R}^N`$ from multiple, possibly motion corrupted, low-resolution stacks of (vectorized) 2D slices $`\vec{y}_k \in\mathbb{R}^{N_k}`$ with $`N_k\ll N`$ for $`k=1,...,\,K`$
for a variety of regularizers $`\text{Reg}`$ and data loss functions $`\varrho`$.
The linear operator $`A_k := D_k B_k W_k`$ represents the combined operator describing the (rigid) motion $`W_k`$, the blurring operator $`B_k`$ and the downsampling operator $`D_k`$.

---

The **available regularizers** include
* Zeroth-order Tikhonov (TK0): $`\text{Reg}(\vec{x}) = \frac{1}{2}\Vert \vec{x} \Vert_{\ell^2}^2`$
* First-order Tikhonov (TK1): $`\text{Reg}(\vec{x}) = \frac{1}{2}\Vert \nabla \vec{x} \Vert_{\ell^2}^2`$
* Isotropic Total Variation (TV): $`\text{Reg}(\vec{x}) = \text{TV}_\text{iso}(\vec{x}) = \big\Vert |\nabla \vec{x}| \big\Vert_{\ell^1}`$
* Huber Function: $`\text{Reg}(\vec{x}) = \frac{1}{2\gamma} \big| |\nabla \vec{x}| \big|_{\gamma}`$

---

The provided **data loss functions** $`\varrho`$ are motivated by [SciPy](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.least_squares.html) and allow for robust outlier rejection. Implemented data loss functions are:
* `linear`: $`\varrho(e) = e `$ 
* `soft_l1`: $`\varrho(e) = 2 (\sqrt{1+e} - 1)`$ 
* `huber`: $`\varrho(e) = |e|_\gamma = \begin{cases} e, & e < \gamma^2 \\ 2\gamma\sqrt{e} - \gamma^2, & e\ge \gamma^2\end{cases}`$
* `arctan`: $`\varrho(e) = \arctan(e)`$
* `cauchy`: $`\varrho(e) = \ln(1 + e)`$

---

Additionally, the choice of finding **optimal reconstruction parameters** is facilitated by the [NumericalSolver](https://cmiclab.cs.ucl.ac.uk/mebner/NumericalSolver) tool.

# Installation
This toolkit depends on a variety of software packages developed within the [GIFT-Surg][giftsurg] including
* [NumericalSolver][numericalsolver]
* [RegistrationTools][registrationtools]
* [PythonHelper][pythonhelper]

Further information on the installation instructions can be found in the [Wiki](https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction/wikis/home).

# Usage

## Volumetric MR Reconstruction from Motion Corrupted 2D Slices
Leveraging a two-step registration-reconstruction approach an isotropic, high-resolution 3D volume can be generated from multiple stacks of low-resolution slices.

Examples for basic usage are:
* `python bin/reconstructVolume.py \
--dir-input dir-with-multiple-stacks \
--dir-output output-dir \
--suffix-mask _mask`
* `python bin/reconstructVolume.py \
--filenames path-to-stack1 ... path-to-stackN \
--dir-output output-dir \
--suffix-mask _mask`

The obtained motion-correction transformations can be stored for further processing, e.g. by using `reconstructVolumeFromSlices.py` to solve the SRR problem for a variety of different regularization and data loss function types.

## SRR Methods for Motion Corrected (or Static) Data

Afer performed motion correction (or having static data in the first place),
2. different solvers and regularizers can be used to solve the SRR problem for comparison, and
1. parameter studies can be performed to find optimal reconstruction parameters.

### SRR from Motion Corrected (or Static) Slice Acquisitions
Solve the SRR problem for motion corrected data:
* `python bin/reconstructVolumeFromSlices.py \
--dir-input dir-to-motion-correction \
--dir-output output-dir \
--reconstruction-type HuberL2 \
--alpha 0.003`
* `python bin/reconstructVolumeFromSlices.py \
--dir-input dir-to-motion-correction \
--dir-output output-dir \
--reconstruction-type TK1L2 \
--alpha 0.03`

Solve the SRR problem for static data:
* `python bin/reconstructVolumeFromSlices.py \
--filenames path-to-stack1 ... path-to-stackN \
--dir-output output-dir \
--reconstruction-type HuberL2 \
--alpha 0.003 
--suffix-mask _mask`
* `python bin/reconstructVolumeFromSlices.py \
--filenames path-to-stack1 ... path-to-stackN \
--dir-output output-dir \
--reconstruction-type TK1L2 \
--alpha 0.03 \
--suffix-mask _mask`

### Parameter Studies to Determine Optimal SRR Parameters
The optimal choice for reconstruction parameters like the regularization parameter or data loss function can be found by running parameter studies. This includes L-curve studies and direct comparison against a reference volume for various cost functions.

Example are:
* `python bin/runReconstructionParameterStudy.py \
--dir-input dir-to-motion-correction \
--reconstruction-type HuberL2 \
--reference path-to-reference-volume.nii.gz \
--reference-mask path-to-reference-volume_mask.nii.gz \
--measures RMSE PSNR NCC NMI SSIM \
--alpha-range 0.001 0.003 10`
* `python bin/runReconstructionParameterStudy.py \
--dir-input dir-to-motion-correction \
--reconstruction-type TVL2 \
--reference path-to-reference-volume.nii.gz \
--reference-mask path-to-reference-volume_mask.nii.gz \
--measures RMSE PSNR NCC NMI SSIM \
--alpha-range 0.001 0.003 10`
* `python bin/runReconstructionParameterStudy.py \
--dir-input dir-to-motion-correction \
--reconstruction-type TK1L2 \
--reference path-to-reference-volume.nii.gz \
--reference-mask path-to-reference-volume_mask.nii.gz \
--measures RMSE PSNR NCC NMI SSIM \
--alpha-range 0.001 0.05 20`

The results can be assessed using the script `showParameterStudy.py` from the [NumericalSolver][numericalsolver] tool.

# License
Copyright (c) 2015-2017, [University College London][ucl].

This framework available as free open-source software under the [BSD-3-Clause License][bsd]. Other licenses may apply for dependencies.


# Funding
This work is partially funded by the UCL [Engineering and Physical Sciences Research Council (EPSRC)][epsrc] Centre for Doctoral Training in Medical Imaging (EP/L016478/1), the Innovative Engineering for Health award ([Wellcome Trust][wellcometrust] [WT101957] and [EPSRC][epsrc] [NS/A000027/1]), and supported by the [National Institute for Health Research][nihr] [University College London Hospitals (UCLH)][uclh] Biomedical Research Centre.

# References
Associated publications are 
* [[Ebner2017]](https://link.springer.com/chapter/10.1007%2F978-3-319-52280-7_1) Ebner, M., Chouhan, M., Patel, P. A., Atkinson, D., Amin, Z., Read, S., Punwani, S., Taylor, S., Vercauteren, T., and Ourselin, S. (2017). Point-Spread-Function-Aware Slice-to-Volume Registration: Application to Upper Abdominal MRI Super-Resolution. In Zuluaga, M. A., Bhatia, K., Kainz, B., Moghari, M. H., and Pace, D. F., editors, Reconstruction, Segmentation, and Analysis of Medical Images. RAMBO 2016, volume 10129 of Lecture Notes in Computer Science, pages 3–13. Springer International Publishing.
* [[Ebner2017a]](https://www.journals.elsevier.com/neuroimage) Ebner, M., Chung, K. K., Prados, F., Cardoso, M. J., Chard, D. T., Vercauteren, T., and Ourselin, S. (In press). Volumetric Reconstruction from Printed Films: Enabling 30 Year Longitudinal Analysis in MR Neuroimaging. NeuroImage.

[citation]: http://www.sciencedirect.com/science/article/pii/S1053811917308042
[mebner]: http://cmictig.cs.ucl.ac.uk/people/phd-students/michael-ebner
[tig]: http://cmictig.cs.ucl.ac.uk
[bsd]: https://opensource.org/licenses/BSD-3-Clause
[giftsurg]: http://www.gift-surg.ac.uk
[cmic]: http://cmic.cs.ucl.ac.uk
[guarantors]: https://guarantorsofbrain.org/
[ucl]: http://www.ucl.ac.uk
[uclh]: http://www.uclh.nhs.uk
[epsrc]: http://www.epsrc.ac.uk
[wellcometrust]: http://www.wellcome.ac.uk
[mssociety]: https://www.mssociety.org.uk/
[nihr]: http://www.nihr.ac.uk/research
[numericalsolver]: https://cmiclab.cs.ucl.ac.uk/mebner/NumericalSolver
[registrationtools]: https://cmiclab.cs.ucl.ac.uk/mebner/RegistrationTools
[pythonhelper]: https://cmiclab.cs.ucl.ac.uk/mebner/PythonHelper