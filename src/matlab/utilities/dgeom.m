function varargout = dgeom(varargin) 
% DGEOM Converts between DICOM IPP, IOP and offcentres and angulations
%
% [iop, ipp] = dgeom(offcentres, angulations, ori, PS_HW , Width, Height)
%    to be deprecated in favour of:
% [iop, ipp] = dgeom(angulations, ori, offcentres, PS_HW , Width, Height)
% [iop] = dgeom(angulations, ori) 
%
% or, DICOM to angulations and offcentres
% [ ang_LPH, ori, off_LPH] = dgeom(iop, ipp, PS_HW , Width, Height)
% [ ang_LPH, ori ] = dgeom(iop)
%
%
% image offcentre (LPH in mm )
% image angulation (LPH in degrees ) NOTE order different from PAR file
% PS_HW Pixel spacing. Two element vector, height then width in mm
% Width Image width (number of pixels)
% Height Image height (number of pixels)
% ori  'TRA', 'COR', or, 'SAG'
% ipp - ImagePositionPatient
% iop - ImageOrientationPatient
%
%
% Image plane is specified by a base orientation with intrinsic
% angulations. Angulations are applied in the order RL-AP-FH which is
% different from the listed order on the UI.
%
% David Atkinson D.Atkinson@ucl.ac.uk
% Uses John Fuller's SpinCalc from the Matlab FileExchange
% See also SpinCalc dgeomextract



LEFT = [1 0 0]; % LPH coordinates
POSTERIOR = [0 1 0];
FOOT = [0 0 -1] ;
HEAD = -FOOT ;

if length(varargin{1})==6
    d2a = true ; % DICOM IOP, IPP  to  offcentres and angulations
else
    d2a = false ; % Angulations and offcentres to DICOM IOP, IPP
end

switch d2a
    case false
        % offcentres and angulations to IOP, IPP
        if nargin > 2 && ischar(varargin{3})
            warning(['Deprecated input order, please see dgeom help.'])
            ofl = varargin{1}(1) ;
            ofp = varargin{1}(2) ;
            ofh = varargin{1}(3) ;
            ang_rad = varargin{2}/180*pi ;
            ori = varargin{3} ;
            PS_HW = varargin{4} ;
            Width = varargin{5} ;
            Height = varargin{6} ;
        else
            ang_rad = varargin{1}/180*pi ;
            ori = varargin{2} ;
            if nargin > 2
                ofl = varargin{3}(1) ;
                ofp = varargin{3}(2) ;
                ofh = varargin{3}(3) ;
                PS_HW = varargin{4} ;
                Width = varargin{5} ;
                Height = varargin{6} ;
            end
        end
        
    case true
        % IOP and IPP to offcentres and angulations
        iop = varargin{1}(:) ; % make it a column vector
        if nargin > 1
            ipp = varargin{2} ;
            PS_HW = varargin{3} ;
            Width = varargin{4} ;
            Height = varargin{5} ;
        end
        
        iop(1:3) = iop(1:3)./norm(iop(1:3)) ;
        iop(4:6) = iop(4:6)./norm(iop(4:6)) ;
        
        snv = cross(iop(1:3),iop(4:6)) ; % slice normal vector
        dptra = dot(snv,cross(LEFT,POSTERIOR)) ; % dot product with normals
        dpcor = dot(snv,cross(LEFT,FOOT)) ;      % of radiological slices
        dpsag = dot(snv,cross(HEAD,POSTERIOR)) ;
        
        [mx, loc] = max([abs(dptra) abs(dpcor) abs(dpsag)]) ;
        switch loc
            case 1
                ori = 'TRA' ;
            case 2
                ori = 'COR' ;
            case 3
                ori = 'SAG' ;
        end
        disp(['Determined ORI: ',ori])
        
end % d2a

% Places origin at centre of image (for even number of pixels, this is at a
% pixel corner. Places IPP at centre of top left voxel, NOT at the corner
% of the pixel.
% With the '-1' this works for axial, sagittal and coronal, and a variety
% of oblique tested on single frame DICOMs from Philips Ingenia (UCLH
% R5.1.7)
if nargin > 2
    ippwoc =  (double(Width-1))/2 * PS_HW(2) ; % ipp width offcentre
    ipphoc =  (double(Height-1))/2 * PS_HW(1) ; % ipp height offcentre
else
    ippwoc = NaN; ipphoc = NaN ; % dummy values
end

% starting image orientation and IOP vectors
switch ori
    case {'tra','TRA','axial','AX','ax','TRANSVERSE','TRANSVERSAL'}
        iops = [LEFT POSTERIOR] ;
        ipps = [-ippwoc -ipphoc 0] ;
    case {'cor','COR','CORONAL'}
        iops = [LEFT FOOT] ;
        ipps = [-ippwoc 0 ipphoc];
    case {'sag','SAG','SAGITTAL'}
        iops = [POSTERIOR FOOT] ;
        ipps = [0 -ippwoc ipphoc] ;
    otherwise
        error(['Orientation not recognised: ',ori])
end

iops = iops(:) ;

switch d2a
    case false
        RL = makehgtform('xrotate',ang_rad(1)) ;
        RP = makehgtform('yrotate',ang_rad(2)) ;
        RH = makehgtform('zrotate',ang_rad(3)) ;
        
        % For rotations, one reverses the order to swap between intrinsic 
        % and extrinsic rotations. 
        % When pre-multiplying a coordinate as a column vector by rotation 
        % matrices, the order is right to left. Hence for the applied 
        % angulation order of RL-AP-FH about rotating (intrinsic) axes, 
        % we use the order:
        iop = RL*RP*RH* cat(2,[iops(1:3);1],[iops(4:6);1]) ;
        iop = iop(1:3,:) ; % remove homog coord
        iop = iop(:) ;
        
        varargout{1} = iop;
        
        if nargin > 2
            Toffc = [ 1 0 0 ofl ;
                      0 1 0 ofp ;
                      0 0 1 ofh ;
                      0 0 0 1 ] ;
            
            %ipp = RL*RP*RH*Toffc* [ipps(:) ; 1] ;
            ipp = Toffc*RL*RP*RH* [ipps(:) ; 1] ;
            ipp = ipp(1:3) ;
            
            varargout{2} = ipp ;
        end
        
    case true
        sns = cross(iops(1:3),iops(4:6)) ;
        
        Ms = cat(2, iops(1:3), iops(4:6), sns(:)) ;
        Md =  cat(2,iop(1:3) ,iop(4:6), snv(:)) ;
        R = Md * inv(Ms) ;
        
        EAHPL = SpinCalc('DCMtoEA321',R,0.01,1) ;
        % Convert from 0-360 to -180 - 180 range
        EAHPL = EAHPL - 360*(EAHPL>180) ;
      
        % Negate signs to correspond to Philips angulations
        ang_LPH = -[EAHPL(3) EAHPL(2) EAHPL(1)] ;
        
        varargout{1} = ang_LPH ;
        varargout{2} = ori ;
        
        if nargin > 1
            ang_LPH_rad = ang_LPH / 360 * 2 * pi ;
            
            % offcentres calculation:
            % We know IPP = Toffc * RL*RP*RH * ipps
            % and that the offcentrea are the right column of
            % Toffc so we can get these elements by subtraction:
            
            RL = makehgtform('xrotate',ang_LPH_rad(1)) ;
            RP = makehgtform('yrotate',ang_LPH_rad(2)) ;
            RH = makehgtform('zrotate',ang_LPH_rad(3)) ;
            vec = RL*RP*RH* [ipps(:) ; 1] ;
            
            offc = ipp(1:3) - vec(1:3) ;
            
            varargout{3} = offc ;
        end
end

end % function





