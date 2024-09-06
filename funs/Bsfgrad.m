%% ***************************************************************
%  filename: Bsfgrad
%% ***************************************************************

function [fobj,grad] = Bsfgrad(Ax,A,gamma,c)

m =size(A,1);

temp_grad=zeros(m,1);

cgamma1 = gamma-c;

cgamma2 = -(c+gamma);

gamma2 = 2*gamma;

tempAx = Ax-cgamma2;

%J1 = find(Ax>-gamma&Ax<0);

%J2 = find(Ax>cgamma1&Ax<=-gamma);

%J3 = find(Ax>=cgamma2&Ax<=cgamma1);

AxJ1 = Ax(Ax>-gamma&Ax<0);

AxJ2 = Ax(Ax>cgamma1&Ax<=-gamma);

tempAxJ3 = tempAx(Ax>=cgamma2&Ax<=cgamma1);

fobj = norm(AxJ1)^2/gamma2-sum(AxJ2)-norm(tempAxJ3)^2/(2*gamma2);

if nargout>=2
      
      temp_grad(Ax>-gamma&Ax<0) = AxJ1/gamma;
      
      temp_grad(Ax>cgamma1&Ax<=-gamma) = -1;
      
      temp_grad(Ax>=cgamma2&Ax<=cgamma1) = tempAxJ3/(gamma2);
      
      grad = A'*temp_grad;
end

