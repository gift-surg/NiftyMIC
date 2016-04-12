function createRectangularMask()

%% Input
%Define range of rectangular mask (use fslview eg)

%Mask for Image FetalNeck/R1/0.nii.gz -- Trachea
% range_x = [100, 165];
% range_y = [120, 170];
% range_z = [6, 14];

%Mask for Image FetalNeck/R2/0.nii.gz -- Brain+Neck
% range_x = [95, 185];
% range_y = [39, 161];
% range_z = [1, 25];
% filename_mask = 'mask_brainneck_rectangular';

%Mask for Image FetalNeck/R2/0.nii.gz -- Respiratory part
% range_x = [85, 140];
% range_y = [110, 175];
% range_z = [4, 13];

%Mask for Image FetalNeck/R2/0.nii.gz -- Trachea
% range_x = [90, 130];
% range_y = [120, 170];
% range_z = [8, 18];

%Mask for Image FetalNeck/R3/0.nii.gz -- Trachea
% range_x = [130, 195];
% range_y = [105, 170];
% range_z = [11, 24];

%Mask for Image FetalNeck/R4/0.nii.gz -- Trachea
% range_x = [100, 170];
% range_y = [117, 183];
% range_z = [8, 16];

%Mask for Image FetalNeck/R6B/0.nii.gz -- Trachea
range_x = [100, 170];
range_y = [95, 165];
range_z = [9, 22];

filename_mask = 'mask_trachea_rectangular';

%% Algorithm
% Find home-folder
if ispc
    home = [getenv('HOMEDRIVE') getenv('HOMEPATH')];
else
    home = getenv('HOME');
end

% dirData = [home '/UCL/Data/Fetal Neck Masses/FETAL_R1 (~)/NIfTI (selected)/'];
% dirData = [home '/UCL/Data/Fetal Neck Masses/FETAL_R2 (+, but cut brains)/NIfTI (selected)/'];
% dirData = [home '/UCL/Data/Fetal Neck Masses/FETAL_R3 (-)/NIfTI (selected, almost all corrupted, new export, nope -- not worth it after view of DICOM images)/'];
% dirData = [home '/UCL/Data/Fetal Neck Masses/FETAL_R4 (+, but no orthogonal views)/NIfTI (selected)/'];
dirData = [home '/UCL/Data/Fetal Neck Masses/FETAL_R6b (--)/NIfTI (selected)/'];
% dirSaveResults = [home '/Desktop/'];
dirSaveResults = dirData;

% Load Image
filename = '0.nii.gz';
filenameShort = filename(1:end-7); %crop ".nii.gz" from filename
nii = load_untouch_nii([dirData filename]);

% Extract image data (intensities)
image_data = nii.img;

% [Nx, Ny, Nz] = size(image_data);
% image_data = image_data(:,end:-1:1,:);

% create mask
mask_data = generateMaskData(image_data, range_x, range_y, range_z);
saveMask(mask_data, nii, dirSaveResults, [filenameShort '_' filename_mask]);


%% Does not work at the moment
%*** show image
% Take care of coordinates! Nx and Ny are intuitive with (1,1) presenting
% lower left corner and (Nx,Ny) presenting upper right.
% BUT: In order to plot it this way with 'imshow' a transposition is needed!
% 
% mean_x = round(sum(range_x)/2);
% mean_y = round(sum(range_y)/2);
% mean_z = round(sum(range_z)/2);
% 
% interval_x = range_x(1):range_x(2);
% interval_y = range_y(1):range_y(2);
% interval_z = range_z(1):range_z(2);
% 
% figure(1)
% clf(1)
% 
% subplot(1,3,1)
% imshow(image_data(interval_x,interval_y,mean_z),[])
% title('original')
% 
% subplot(1,3,2)
% imshow(image_data(interval_x,mean_y,interval_z),[])
% title('mask')
% 
% subplot(1,3,3)
% imshow(image_data(mean_x,interval_y,interval_z),[])
% title('masked image')

end

%%
function mask_data = generateMaskData(image_data, range_x, range_y, range_z)
mask_data = zeros(size(image_data));

% [Nx, Ny, Nz] = size(image_data);

interval_x = range_x(1):range_x(2);
interval_y = range_y(1):range_y(2);
interval_z = range_z(1):range_z(2);

mask_data(interval_x, interval_y, interval_z) = 1;
end

%%
function mask = saveMask(data, nii, directory, filename)

nii.img = data;
save_untouch_nii(nii,[directory filename '.nii.gz'])
end