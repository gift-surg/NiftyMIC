#!/bin/sh

python reconstructRestingStateVolume.py \
	--ADMM-iterations=10 \
	--alpha=0.1 \
	--alpha-final=0.03 \
	--bias-field-correction=0 \
	--data-loss=linear \
	--dilation-radius=3 \
	--dir-output=results/ \
	--extra-frame-target=10 \
	--filename=/Users/mebner/UCL/Data/rsfmri/bold.nii.gz \
	--filename-mask=/Users/mebner/UCL/Data/rsfmri/bold_4Dmask.nii.gz \
	--intensity-correction=0 \
	--isotropic-resolution=None \
	--iter-max=5 \
	--iter-max-final=10 \
	--log-motion-correction=1 \
	--log-script-execution=1 \
	--minimizer=lsmr \
	--prefix-output=SRR_ \
	--provide-comparison=1 \
	--regularization=TK1 \
	--rho=0.5 \
	--sigma=0.9 \
	--stack-recon-range=3 \
	--target-stack-index=0 \
	--two-step-cycles=3 \
	--verbose=1 \
