function test_geom
% TEST_GEOM Used for testing dgeom conversions between DICOM and Philips
%
%
% D.Atkinson@ucl.ac.uk
% See also DGEOM

sfn = pref_uigetfile('test_geom','sf') ;

dinfo = dicominfo(sfn) ;

ang_AP = dinfo.Private_2005_1000 ;
ang_FH = dinfo.Private_2005_1001 ;
ang_RL = dinfo.Private_2005_1002 ;

disp(['ang AP / FH / RL ',num2str(ang_AP),' / ',num2str(ang_FH), ' / ', ...
    num2str(ang_RL)])

ang_lph = [ang_RL ang_AP ang_FH] ;

offc_AP = dinfo.Private_2005_1008 ;
offc_FH = dinfo.Private_2005_1009 ;
offc_RL = dinfo.Private_2005_100a ;

offc_lph = [offc_RL offc_AP offc_FH];

ORI = dinfo.Private_2001_100b ;

PS = dinfo.PixelSpacing ;
Width = dinfo.Width ;
Height = dinfo.Height ;

iop_act = dinfo.ImageOrientationPatient ;
ipp_act = dinfo.ImagePositionPatient ;

[iop, ipp] = dgeom(offc_lph, ang_lph, ORI, PS , Width, Height) ;

disp(['IOP file    comp'])
iop_disp = cat(2,iop_act(:), iop(:))

disp(['IPP file    comp'])
ipp_disp = cat(2, ipp_act(:), ipp(:))


[ang_LPH, ori, offc] = dgeom(iop_act, ipp_act, PS, Width, Height) ;
disp(['Computed ang: P / H / L ', num2str(ang_LPH(2)),' / ', num2str(ang_LPH(3)), ...
    ' / ', num2str(ang_LPH(1))])
disp(['File ORI: ',ORI, ' computed ori: ',ori])
disp(['Computed offcentres: ',num2str(offc(:)')])
disp(['File offcentres: ',num2str([offc_RL offc_AP offc_FH])])





