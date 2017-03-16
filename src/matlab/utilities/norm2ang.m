function norm2ang
% NORM2ANG Plane normals to angulations (Philips)
% Compute the oblique directions used to define the oblique acquisition
% planes
% Author: David Atkinson, received 6 Feb 2017

%% Corners of cubes (will be normalised later)

%*** Lower half of cube (seems to work)
p(1).normal = [ 1  1 -1] ;
p(2).normal = [-1  1 -1] ;
p(3).normal = [ 1 -1 -1] ;
p(4).normal = [-1 -1 -1] ;

%*** Left half of cube
% p(1).normal = [ 1  -1 1] ;
% p(2).normal = [-1  -1 1] ;
% p(3).normal = [ 1 -1 -1] ;
% p(4).normal = [-1 -1 -1] ;

% Pick in-plane vectors by finding one that is perpendicular to both the
% normal and, say, [0 1 0]

refv = [ 0 1 0 ];

for ip = 1:length(p)
    
    % Normalize vector pointing to cube corner
    nnorm = norm(p(ip).normal) ;
    p(ip).normal = p(ip).normal ./ nnorm ;
    
    % Get orthonormal vector w.r.t to reference normal refv
    % v1u = normal x refv
    p(ip).v1u = cross(p(ip).normal, refv) ;
    p(ip).v1u = p(ip).v1u ./ norm(p(ip).v1u) ;
    
    % v2u = normal x v1u = normal x (normal x refv)
    %     = normal (normal dot refv) - refv (normal dot normal)
    %     = normal (normal dot refv) - refv 
    p(ip).v2u = cross(p(ip).normal, p(ip).v1u) ;
    
    % Define image orientation plane (iop)
    iop = [p(ip).v1u, -p(ip).v2u];
    
    [ang_LPH, ORI] = dgeom(iop)
end


