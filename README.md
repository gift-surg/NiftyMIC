# Volumetric MRI Reconstruction from Motion Corrupted 2D Slices

This is a research-focused toolkit developed within the [GIFT-Surg](http://www.gift-surg.ac.uk/) project to reconstruct an isotropic, high-resolution volume from multiple, possibly motion-corrupted, stacks of low-resolution 2D slices. The framework relies on slice-to-volume registration algorithms for motion correction and reconstruction-based Super-Resolution (SR) techniques for the volumetric reconstruction. 
The entire reconstruction pipeline is programmed in Python by using a mix of SimpleITK, WrapITK and standard C++ITK.

If you have any questions or comments (or find bugs), please drop an email to @mebner (`michael.ebner.14@ucl.ac.uk`).

# How it works

Several methods have been implemented to solve the SR Reconstruction (SRR) problem
$\vec{x^*} := \argmin_x \Big[\sum_{k=1}^K \sum_{i=1}^{N_k} \varrho\big( (A_k\vec{x} - \vec{y}_k)_i^2 \big) + \alpha \text{Reg}(\vec{x}) \Big]$

to obtain the high-resolution volume $\vec{x}$ from multiple, possibly (rigid) motion corrupted, low-resolution stacks of 2D slices ${\vec{y}_k}_{k=1}^K$
for a variety of regularizers $\Reg$ and data loss functions $\varrho$.
Regularizers include
* Zeroth-order Tikhonov: $\text{Reg}(\vec{x}) = \frac{1}{2}\Vert \vec{x} \Vert_{\ell^2}^2$
* First-order Tikhonov: $\text{Reg}(\vec{x}) = \frac{1}{2}\Vert \nabla \vec{x} \Vert_{\ell^2}^2$
* Isotropic Total Variation: $\text{Reg}(\vec{x}) = \text{TV}_\text{iso}(\vec{x}) = \Vert |\nabla \vec{x}| \Vert_{\ell^1}$
* Huber Function: $\text{Reg}(\vec{x}) =  \big| |\nabla \vec{x}| \big|_{\gamma}$

Data loss functions $\varrho$ are motivated by [[SciPy]](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.least_squares.html) and include 
* `linear`: $\varrho(e) = e $ 
* `soft_l1`: $\varrho(e) = 2 (\sqrt{1+e} - 1)$ 
<!-- * `huber`: $\varrho(e) = 2 (\sqrt{1+e} - 1)$  -->
* `arctan`: $\varrho(e) = \text{arctan}(e)$
* `cauchy`: $\varrho(e) = \text{cauchy}(e)$

# Installation
The Installation instructions can be found in the [[Wiki]](https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction/wikis/home).

# Usage

## Volumetric MR Reconstruction from Motion Corrupted 2D Slices
Leveraging a two-step registration-reconstruction approach an isotropic, high-resolution 3D volume can be generated from multiple stacks of low-resolution slices.

Examples for basic usage are:
* `python bin/reconstructVolume.py --dir-input dir-with-multiple-stacks --dir-output output-dir`
* `python bin/reconstructVolume.py --filenames path-to-stack1 ... path-to-stackN --dir-output output-dir`

The obtained motion-correction transformations can be stored for further processing, e.g. for `reconstructVolumeFromSlices.py` to solve the SRR problem for a variety of different solvers.

## Super-Resolution Reconstruction Methods for Motion Corrected (or Static) Data

The SRR problem can be solved for a variety of different solvers for
..1. motion corrected data and, 
..2. static data

### Super-Resolution Reconstruction from Motion Corrected (or Static) Slice Acquisitions
Solve the SRR problem for motion corrected data:
* `python bin/reconstructVolumeFromSlices.py --dir-input dir-to-motion-correction --dir-output output-dir --reconstruction-type TVL2 --alpha 0.003`

Solve the SRR problem for static data:
* `python bin/reconstructVolumeFromSlices.py --filenames path-to-stack1 ... path-to-stackN --dir-output output-dir --reconstruction-type HuberL2 --alpha 0.003`
* `python bin/reconstructVolumeFromSlices.py --filenames path-to-stack1 ... path-to-stackN --dir-output output-dir --reconstruction-type HuberL2 --alpha 0.003 --suffix-mask _mask`

### Parameter Studies to Determine Optimal Reconstruction Parameters
The optimal choice for reconstruction parameters like the regularization parameter or data loss function can be found by running parameter studies.

Examples for usage are:
* `python bin/runReconstructionParameterStudy.py --dir-input dir-to-motion-correction --reconstruction-type TVL2 --alpha-range 0.001 0.003 10`
* `python bin/runReconstructionParameterStudy.py --dir-input dir-to-motion-correction --reconstruction-type TVL2 --alpha-range 0.001 0.003 10`
* `python bin/runReconstructionParameterStudy.py --dir-input dir-to-motion-correction --reconstruction-type TK1L2 --alpha-range 0.001 0.05 20 --data-loss arctan --data-loss-scale 0.8`


# References
Associated publications are 
* [[Ebner2017]](https://link.springer.com/chapter/10.1007%2F978-3-319-52280-7_1) Ebner, M., Chouhan, M., Patel, P. A., Atkinson, D., Amin, Z., Read, S., Punwani, S., Taylor, S., Vercauteren, T., and Ourselin, S. (2017). Point-Spread-Function-Aware Slice-to-Volume Registration: Application to Upper Abdominal MRI Super-Resolution. In Zuluaga, M. A., Bhatia, K., Kainz, B., Moghari, M. H., and Pace, D. F., editors, Reconstruction, Segmentation, and Analysis of Medical Images. RAMBO 2016, volume 10129 of Lecture Notes in Computer Science, pages 3–13. Springer International Publishing.
* [[Ebner2017a]](https://www.journals.elsevier.com/neuroimage) Ebner, M., Chung, K. K., Prados, F., Cardoso, M. J., Chard, D. T., Vercauteren, T., and Ourselin, S. (In press). Volumetric Reconstruction from Printed Films: Enabling 30 Year Longitudinal Analysis in MR Neuroimaging. NeuroImage.


# License
This toolkit is still under development and has NOT been publicly released yet.
See LICENSE file for details.
