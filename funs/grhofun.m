%% ****************************************************************
%  filename: grhofun
%% ****************************************************************
% grho(x) = (1/rho)*sum_{i=1}^n psi^*(rho|x_i|)
%

function [obj,grad] = grhofun(x,rho,acon)

p = size(x,1);

temp_obj = zeros(p,1);

temp_grad = zeros(p,1);

acon0 = acon+1;

acon2 = 2*acon/acon0;

acon1 = 2/acon0;

rho_xabs = rho*abs(x);

J1 = find((rho_xabs>=acon2));

J2 = find((rho_xabs>=acon1)&(rho_xabs<acon2));

temp_obj(J1) = rho_xabs(J1)-1;

tempv = (acon0*rho_xabs(J2)-2)/(2*(acon-1));

temp_obj(J2) =((acon-1)/acon0)*tempv.^2;

obj = (1/rho)*sum(temp_obj);

if nargout>=2
    
    temp_grad(J1) = 1;
    
    temp_grad(J2) = tempv;
    
    grad = temp_grad.*sign(x);
end
end