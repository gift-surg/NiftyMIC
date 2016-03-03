
addpath('/Users/mebner/development/Matlab/phantom3d')

% dimension of output image, i.e. NxNxN voxels
N = 64;

dir_output = '/tmp/';
% dir_output = '/Users/mebner/UCL/UCL/Volumetric Reconstruction/GettingStarted/data/';

filename_output = ['3D_SheppLoganPhantom_' num2str(N) '.nii.gz'];

% get data array scaled from 0 to 255
data = round(255*phantom3d('Modified Shepp-Logan', N));

% create nifti file
nii = make_nii(data);

% view_nii(nii)

% save nifti file
save_nii(nii, [dir_output filename_output]);

% view saved nifti file via itksnap
system(['itksnap ' dir_output filename_output])