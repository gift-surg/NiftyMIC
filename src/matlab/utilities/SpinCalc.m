function OUTPUT=SpinCalc(CONVERSION,INPUT,tol,ichk)
%Function for the conversion of one rotation input type to desired output.
%Supported conversion input/output types are as follows:
%   1: Q        Rotation Quaternions
%   2: EV       Euler Vector and rotation angle (degrees)
%   3: DCM      Orthogonal DCM Rotation Matrix
%   4: EA###    Euler angles (12 possible sets) (degrees)
%
%Author: John Fuller
%National Institute of Aerospace
%Hampton, VA 23666
%John.Fuller@nianet.org
%
%Version 1.3
%June 30th, 2009
%
%Version 1.3 updates
%   SpinCalc now detects when input data is too close to Euler singularity, if user is choosing
%   Euler angle output. Prohibits output if middle angle is within 0.1 degree of singularity value.
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%                OUTPUT=SpinCalc(CONVERSION,INPUT,tol,ichk)
%Inputs:
%CONVERSION - Single string value that dictates the type of desired
%             conversion.  The conversion strings are listed below.
%
%   'DCMtoEA###'  'DCMtoEV'    'DCMtoQ'       **for cases that involve   
%   'EA###toDCM'  'EA###toEV'  'EA###toQ'       euler angles, ### should be
%   'EVtoDCM'     'EVtoEA###'  'EVtoQ'          replaced with the proper 
%   'QtoDCM'      'QtoEA###'   'QtoEV'          order desired.  EA321 would
%   'EA###toEA###'                              be Z(yaw)-Y(pitch)-X(roll).
%
%INPUT - matrix or vector that corresponds to the first entry in the
%        CONVERSION string, formatted as follows:
%
%        DCM - 3x3xN multidimensional matrix which pre-multiplies a coordinate
%              frame column vector to calculate its coordinates in the desired 
%              new frame.
%
%        EA### - [psi,theta,phi] (Nx3) row vector list dictating to the first angle
%                rotation (psi), the second (theta), and third (phi) (DEGREES)
%
%        EV - [m1,m2,m3,MU] (Nx4) row vector list dictating the components of euler
%             rotation vector (original coordinate frame) and the Euler 
%             rotation angle about that vector (MU) (DEGREES)
%
%        Q - [q1,q2,q3,q4] (Nx4) row vector list defining quaternion of
%            rotation.  q4 = cos(MU/2) where MU is Euler rotation angle
%
%tol - tolerance value
%ichk - 0 disables warning flags
%          1 enables warning flags (near singularities)
%**NOTE: N corresponds to multiple orientations
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%Output:
%OUTPUT - matrix or vector corresponding to the second entry in the
%         CONVERSION input string, formatted as shown above.
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%Pre-processer to determine type of conversion from CONVERSION string input
%Types are numbered as follows:
%Q=1   EV=2   DCM=3   EA=4
i_type=strfind(lower(CONVERSION),'to');
length=size(CONVERSION,2);
if length>12 || length<4,   %no CONVERSION string can be shorter than 4 or longer than 12 chars
    error('Error: Invalid entry for CONVERSION input string');
end
o_type=length-i_type;
if i_type<5,
    i_type=i_type-1;
else
    i_type=i_type-2;
end
if o_type<5,
    o_type=o_type-1;
else
    o_type=o_type-2;
end
TYPES=cell(1,4);
TYPES{1,1}='Q'; TYPES{1,2}='EV'; TYPES{1,3}='DCM'; TYPES{1,4}='EA';
INPUT_TYPE=TYPES{1,i_type};
OUTPUT_TYPE=TYPES{1,o_type};
clear TYPES
%Confirm input as compared to program interpretation
if i_type~=4 && o_type~=4,  %if input/output are NOT Euler angles
    CC=[INPUT_TYPE,'to',OUTPUT_TYPE];
    if strcmpi(CONVERSION,CC)==0;
        error('Error: Invalid entry for CONVERSION input string');        
    end
else
    if i_type==4,   %if input type is Euler angles, determine the order of rotations
        EULER_order_in=str2double(CONVERSION(1,3:5));
        rot_1_in=floor(EULER_order_in/100);     %first rotation
        rot_2_in=floor((EULER_order_in-rot_1_in*100)/10);   %second rotation
        rot_3_in=(EULER_order_in-rot_1_in*100-rot_2_in*10);   %third rotation
        if rot_1_in<1 || rot_2_in<1 || rot_3_in<1 || rot_1_in>3 || rot_2_in>3 || rot_3_in>3,
            error('Error: Invalid input Euler angle order type (conversion string).');  %check that all orders are between 1 and 3            
        elseif rot_1_in==rot_2_in || rot_2_in==rot_3_in,
            error('Error: Invalid input Euler angle order type (conversion string).');  %check that no 2 consecutive orders are equal (invalid)           
        end
        %check input dimensions to be 1x3x1
        if size(INPUT,2)~=3 || size(INPUT,3)~=1
            error('Error: Input euler angle data vector is not Nx3')            
        end
        %identify singularities        
        if rot_1_in==rot_3_in, %Type 2 rotation (first and third rotations about same axis)
            if INPUT(:,2)<=zeros(size(INPUT,1),1) | INPUT(:,2)>=180*ones(size(INPUT,1),1),  %confirm second angle within range
                error('Error: Second input Euler angle(s) outside 0 to 180 degree range')
            elseif abs(INPUT(:,2))<2*ones(size(INPUT,1),1) | abs(INPUT(:,2))>178*ones(size(INPUT,1),1),  %check for singularity
                if ichk==1,
                    errordlg('Warning: Input Euler angle rotation(s) near a singularity.               Second angle near 0 or 180 degrees.')
                end
            end
        else    %Type 1 rotation (all rotations about each of three axes)
            if abs(INPUT(:,2))>=90*ones(size(INPUT,1),1), %confirm second angle within range
                error('Error: Second input Euler angle(s) outside -90 to 90 degree range')
            elseif abs(INPUT(:,2))>88*ones(size(INPUT,1),1),  %check for singularity
                if ichk==1,
                    errordlg('Warning: Input Euler angle(s) rotation near a singularity.               Second angle near -90 or 90 degrees.')
                end
            end
        end    
    end
    if o_type==4,   %if output type is Euler angles, determine order of rotations
        EULER_order_out=str2double(CONVERSION(1,length-2:length));
        rot_1_out=floor(EULER_order_out/100);   %first rotation
        rot_2_out=floor((EULER_order_out-rot_1_out*100)/10);    %second rotation
        rot_3_out=(EULER_order_out-rot_1_out*100-rot_2_out*10); %third rotation
        if rot_1_out<1 || rot_2_out<1 || rot_3_out<1 || rot_1_out>3 || rot_2_out>3 || rot_3_out>3,
            error('Error: Invalid output Euler angle order type (conversion string).'); %check that all orders are between 1 and 3           
        elseif rot_1_out==rot_2_out || rot_2_out==rot_3_out,
            error('Error: Invalid output Euler angle order type (conversion string).'); %check that no 2 consecutive orders are equal
        end
    end        
    if i_type==4 && o_type~=4,  %if input are euler angles but not output
        CC=['EA',num2str(EULER_order_in),'to',OUTPUT_TYPE]; %construct program conversion string for checking against user input
    elseif o_type==4 && i_type~=4,  %if output are euler angles but not input
        CC=[INPUT_TYPE,'to','EA',num2str(EULER_order_out)]; %construct program conversion string for checking against user input
    elseif i_type==4 && o_type==4,  %if both input and output are euler angles
        CC=['EA',num2str(EULER_order_in),'to','EA',num2str(EULER_order_out)];   %construct program conversion string
    end
    if strcmpi(CONVERSION,CC)==0; %check program conversion string against user input to confirm the conversion command
        error('Error: Invalid entry for CONVERSION input string');
    end
end
clear i_type o_type CC

%From the input, determine the quaternions that uniquely describe the
%rotation prescribed by that input.  The output will be calculated in the
%second portion of the code from these quaternions.
switch INPUT_TYPE
    case 'DCM'
        if size(INPUT,1)~=3 || size(INPUT,2)~=3  %check DCM dimensions
            error('Error: DCM matrix is not 3x3xN');           
        end
        N=size(INPUT,3);    %number of orientations
        %Check if matrix is indeed orthogonal
        perturbed=NaN(3,3,N);
        DCM_flag=0;
        for ii=1:N,
            perturbed(:,:,ii)=abs(INPUT(:,:,ii)*INPUT(:,:,ii)'-eye(3)); %perturbed array shows difference between DCM*DCM' and I
            if abs(det(INPUT(:,:,ii))-1)>tol, %if determinant is off by one more than tol, user is warned.
                if ichk==1,
                    DCM_flag=1;
                end
            end
            if abs(det(INPUT(:,:,ii))+1)<0.05, %if determinant is near -1, DCM is improper
                error('Error: Input DCM(s) improper');           
            end
            if DCM_flag==1,
                errordlg('Warning: Input DCM matrix determinant(s) off from 1 by more than tolerance.')
            end
        end
        DCM_flag=0;
        if ichk==1,
            for kk=1:N,
                for ii=1:3,
                    for jj=1:3,
                        if perturbed(ii,jj,kk)>tol,   %if any difference is larger than tol, user is warned.
                            DCM_flag=1;
                        end
                    end
                end
            end
            if DCM_flag==1,
                fprintf('Warning: Input DCM(s) matrix not orthogonal to precision tolerance.')
            end
        end       
        clear perturbed DCM_flag   
        Q=NaN(4,N);
        for ii=1:N,
            denom=NaN(4,1);
            denom(1)=0.5*sqrt(1+INPUT(1,1,ii)-INPUT(2,2,ii)-INPUT(3,3,ii));
            denom(2)=0.5*sqrt(1-INPUT(1,1,ii)+INPUT(2,2,ii)-INPUT(3,3,ii));
            denom(3)=0.5*sqrt(1-INPUT(1,1,ii)-INPUT(2,2,ii)+INPUT(3,3,ii));
            denom(4)=0.5*sqrt(1+INPUT(1,1,ii)+INPUT(2,2,ii)+INPUT(3,3,ii));        
            %determine which Q equations maximize denominator
            switch find(denom==max(denom),1,'first')  %determines max value of qtests to put in denominator
                case 1
                    Q(1,ii)=denom(1);
                    Q(2,ii)=(INPUT(1,2,ii)+INPUT(2,1,ii))/(4*Q(1,ii));
                    Q(3,ii)=(INPUT(1,3,ii)+INPUT(3,1,ii))/(4*Q(1,ii));
                    Q(4,ii)=(INPUT(2,3,ii)-INPUT(3,2,ii))/(4*Q(1,ii));
                case 2
                    Q(2,ii)=denom(2);
                    Q(1,ii)=(INPUT(1,2,ii)+INPUT(2,1,ii))/(4*Q(2,ii));
                    Q(3,ii)=(INPUT(2,3,ii)+INPUT(3,2,ii))/(4*Q(2,ii));
                    Q(4,ii)=(INPUT(3,1,ii)-INPUT(1,3,ii))/(4*Q(2,ii));
                case 3
                    Q(3,ii)=denom(3);
                    Q(1,ii)=(INPUT(1,3,ii)+INPUT(3,1,ii))/(4*Q(3,ii));
                    Q(2,ii)=(INPUT(2,3,ii)+INPUT(3,2,ii))/(4*Q(3,ii));
                    Q(4,ii)=(INPUT(1,2,ii)-INPUT(2,1,ii))/(4*Q(3,ii));
                case 4
                    Q(4,ii)=denom(4);
                    Q(1,ii)=(INPUT(2,3,ii)-INPUT(3,2,ii))/(4*Q(4,ii));
                    Q(2,ii)=(INPUT(3,1,ii)-INPUT(1,3,ii))/(4*Q(4,ii));
                    Q(3,ii)=(INPUT(1,2,ii)-INPUT(2,1,ii))/(4*Q(4,ii));
            end
        end
        Q=Q';
        clear denom
    case 'EV'  %Euler Vector Input Type
        if size(INPUT,2)~=4 || size(INPUT,3)~=1   %check dimensions
            error('Error: Input euler vector and rotation data matrix is not Nx4')            
        end
        N=size(INPUT,1);
        MU=INPUT(:,4)*pi/180;  %assign mu name for clarity
        if sqrt(INPUT(:,1).^2+INPUT(:,2).^2+INPUT(:,3).^2)-ones(N,1)>tol*ones(N,1),  %check that input m's constitute unit vector
            error('Input euler vector(s) components do not constitute a unit vector')            
        end
        if MU<zeros(N,1) | MU>2*pi*ones(N,1), %check if rotation about euler vector is between 0 and 360
            error('Input euler rotation angle(s) not between 0 and 360 degrees')
        end
        Q=[INPUT(:,1).*sin(MU/2),INPUT(:,2).*sin(MU/2),INPUT(:,3).*sin(MU/2),cos(MU/2)];   %quaternion
        clear m1 m2 m3 MU
    case 'EA'        
        psi=INPUT(:,1)*pi/180;  theta=INPUT(:,2)*pi/180;  phi=INPUT(:,3)*pi/180;
        N=size(INPUT,1);    %number of orientations
        %Pre-calculate cosines and sines of the half-angles for conversion.
        c1=cos(psi./2); c2=cos(theta./2); c3=cos(phi./2);
        s1=sin(psi./2); s2=sin(theta./2); s3=sin(phi./2);
        c13=cos((psi+phi)./2);  s13=sin((psi+phi)./2);
        c1_3=cos((psi-phi)./2);  s1_3=sin((psi-phi)./2);
        c3_1=cos((phi-psi)./2);  s3_1=sin((phi-psi)./2);
        if EULER_order_in==121,
            Q=[c2.*s13,s2.*c1_3,s2.*s1_3,c2.*c13];
        elseif EULER_order_in==232,
            Q=[s2.*s1_3,c2.*s13,s2.*c1_3,c2.*c13];
        elseif EULER_order_in==313;
            Q=[s2.*c1_3,s2.*s1_3,c2.*s13,c2.*c13];
        elseif EULER_order_in==131,
            Q=[c2.*s13,s2.*s3_1,s2.*c3_1,c2.*c13];
        elseif EULER_order_in==212,
            Q=[s2.*c3_1,c2.*s13,s2.*s3_1,c2.*c13];
        elseif EULER_order_in==323,
            Q=[s2.*s3_1,s2.*c3_1,c2.*s13,c2.*c13];
        elseif EULER_order_in==123,
            Q=[s1.*c2.*c3+c1.*s2.*s3,c1.*s2.*c3-s1.*c2.*s3,c1.*c2.*s3+s1.*s2.*c3,c1.*c2.*c3-s1.*s2.*s3];
        elseif EULER_order_in==231,
            Q=[c1.*c2.*s3+s1.*s2.*c3,s1.*c2.*c3+c1.*s2.*s3,c1.*s2.*c3-s1.*c2.*s3,c1.*c2.*c3-s1.*s2.*s3];
        elseif EULER_order_in==312,
            Q=[c1.*s2.*c3-s1.*c2.*s3,c1.*c2.*s3+s1.*s2.*c3,s1.*c2.*c3+c1.*s2.*s3,c1.*c2.*c3-s1.*s2.*s3];
        elseif EULER_order_in==132,
            Q=[s1.*c2.*c3-c1.*s2.*s3,c1.*c2.*s3-s1.*s2.*c3,c1.*s2.*c3+s1.*c2.*s3,c1.*c2.*c3+s1.*s2.*s3];
        elseif EULER_order_in==213,
            Q=[c1.*s2.*c3+s1.*c2.*s3,s1.*c2.*c3-c1.*s2.*s3,c1.*c2.*s3-s1.*s2.*c3,c1.*c2.*c3+s1.*s2.*s3];
        elseif EULER_order_in==321,
            Q=[c1.*c2.*s3-s1.*s2.*c3,c1.*s2.*c3+s1.*c2.*s3,s1.*c2.*c3-c1.*s2.*s3,c1.*c2.*c3+s1.*s2.*s3];
        else
            error('Error: Invalid input Euler angle order type (conversion string)');            
        end
        clear c1 s1 c2 s2 c3 s3 c13 s13 c1_3 s1_3 c3_1 s3_1 psi theta phi
    case 'Q'
        if size(INPUT,2)~=4 || size(INPUT,3)~=1
            error('Error: Input quaternion matrix is not Nx4');            
        end
        N=size(INPUT,1);    %number of orientations 
        if ichk==1,
            if abs(sqrt(INPUT(:,1).^2+INPUT(:,2).^2+INPUT(:,3).^2+INPUT(:,4).^2)-ones(N,1))>tol*ones(N,1)
                errordlg('Warning: Input quaternion norm(s) deviate(s) from unity by more than tolerance')
            end 
        end
        Q=INPUT;
end
clear INPUT INPUT_TYPE EULER_order_in

%Normalize quaternions in case of deviation from unity.  User has already
%been warned of deviation.
Qnorms=sqrt(sum(Q.*Q,2));
Q=[Q(:,1)./Qnorms,Q(:,2)./Qnorms,Q(:,3)./Qnorms,Q(:,4)./Qnorms];

switch OUTPUT_TYPE
    case 'DCM'
        Q=reshape(Q',1,4,N);
        OUTPUT=[Q(1,1,:).^2-Q(1,2,:).^2-Q(1,3,:).^2+Q(1,4,:).^2,2*(Q(1,1,:).*Q(1,2,:)+Q(1,3,:).*Q(1,4,:)),2*(Q(1,1,:).*Q(1,3,:)-Q(1,2,:).*Q(1,4,:));
                2*(Q(1,1,:).*Q(1,2,:)-Q(1,3,:).*Q(1,4,:)),-Q(1,1,:).^2+Q(1,2,:).^2-Q(1,3,:).^2+Q(1,4,:).^2,2*(Q(1,2,:).*Q(1,3,:)+Q(1,1,:).*Q(1,4,:));
                2*(Q(1,1,:).*Q(1,3,:)+Q(1,2,:).*Q(1,4,:)),2*(Q(1,2,:).*Q(1,3,:)-Q(1,1,:).*Q(1,4,:)),-Q(1,1,:).^2-Q(1,2,:).^2+Q(1,3,:).^2+Q(1,4,:).^2];
    case 'EV'
        MU=2*atan2(sqrt(sum(Q(:,1:3).*Q(:,1:3),2)),Q(:,4));
        if sin(MU/2)~=zeros(N,1),
            OUTPUT=[Q(:,1)./sin(MU/2),Q(:,2)./sin(MU/2),Q(:,3)./sin(MU/2),MU*180/pi];
        else
            OUTPUT=NaN(N,4);
            for ii=1:N,
                if sin(MU(ii,1)/2)~=0,
                    OUTPUT(ii,1:4)=[Q(ii,1)/sin(MU(ii,1)/2),Q(ii,2)/sin(MU(ii,1)/2),Q(ii,3)/sin(MU(ii,1)/2),MU(ii,1)*180/pi];
                else
                    OUTPUT(ii,1:4)=[1,0,0,MU(ii,1)*180/pi];
                end
            end
        end
    case 'Q'
        OUTPUT=Q;
    case 'EA'
        if EULER_order_out==121,
            psi=atan2((Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)));
            theta=acos(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,2)-Q(:,3).*Q(:,4)),(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
	  Euler_type=2;
        elseif EULER_order_out==232;
            psi=atan2((Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)),(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)));
            theta=acos(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2);
            phi=atan2((Q(:,2).*Q(:,3)-Q(:,1).*Q(:,4)),(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)));
	  Euler_type=2;
        elseif EULER_order_out==313;
            psi=atan2((Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)));
            theta=acos(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,3)-Q(:,2).*Q(:,4)),(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)));
	  Euler_type=2;
        elseif EULER_order_out==131;
            psi=atan2((Q(:,1).*Q(:,3)-Q(:,2).*Q(:,4)),(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)));
            theta=acos(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)));
	  Euler_type=2;
        elseif EULER_order_out==212;
            psi=atan2((Q(:,1).*Q(:,2)-Q(:,3).*Q(:,4)),(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)));
            theta=acos(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)));
	  Euler_type=2;
        elseif EULER_order_out==323;
            psi=atan2((Q(:,2).*Q(:,3)-Q(:,1).*Q(:,4)),(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
            theta=acos(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2);
            phi=atan2((Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)),(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)));
	  Euler_type=2;
        elseif EULER_order_out==123;
            psi=atan2(2.*(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
            theta=asin(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
            phi=atan2(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==231;
            psi=atan2(2.*(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
            theta=asin(2.*(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)));
            phi=atan2(2.*(Q(:,1).*Q(:,4)-Q(:,3).*Q(:,2)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==312;
            psi=atan2(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
            theta=asin(2.*(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)));
            phi=atan2(2.*(Q(:,2).*Q(:,4)-Q(:,3).*Q(:,1)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==132;
            psi=atan2(2.*(Q(:,1).*Q(:,4)+Q(:,2).*Q(:,3)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
            theta=asin(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)));
            phi=atan2(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==213;
            psi=atan2(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
            theta=asin(2.*(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)));
            phi=atan2(2.*(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,4).^2-Q(:,1).^2+Q(:,2).^2-Q(:,3).^2));
	  Euler_type=1;
        elseif EULER_order_out==321;
            psi=atan2(2.*(Q(:,1).*Q(:,2)+Q(:,3).*Q(:,4)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
            theta=asin(2.*(Q(:,2).*Q(:,4)-Q(:,1).*Q(:,3)));
            phi=atan2(2.*(Q(:,1).*Q(:,4)+Q(:,3).*Q(:,2)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
	  Euler_type=1;
        else
            error('Error: Invalid output Euler angle order type (conversion string).');           
        end
        if(isreal([psi,theta,phi]))==0,
	        error('Error: Unreal Euler output.  Input resides too close to singularity.  Please choose different output type.')
        end
        OUTPUT=mod([psi,theta,phi]*180/pi,360);  %deg
        if Euler_type==1,
	        sing_chk=find(abs(theta)*180/pi>89.9);
	        sing_chk=sort(sing_chk(sing_chk>0));
	        if size(sing_chk,1)>=1,
		        error('Error: Input rotation #%s resides too close to Type 1 Euler singularity.\nType 1 Euler singularity occurs when second angle is -90 or 90 degrees.\nPlease choose different output type.',num2str(sing_chk(1,1)));
	        end
        elseif Euler_type==2,
	        sing_chk=[find(abs(theta*180/pi)<0.1);find(abs(theta*180/pi-180)<0.1);find(abs(theta*180/pi-360))<0.1];
	        sing_chk=sort(sing_chk(sing_chk>0));
	        if size(sing_chk,1)>=1,
		        error('Error: Input rotation #%s resides too close to Type 2 Euler singularity.\nType 2 Euler singularity occurs when second angle is 0 or 180 degrees.\nPlease choose different output type.',num2str(sing_chk(1,1)));
	        end
        end
end






