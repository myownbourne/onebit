%% ***************************************************************
%  filename: Bsfgrad_scad
%% ***************************************************************

function [fobj,grad] = Bsfgrad_scad(x,Ax,A,gamma,rho,acon,lambda_rho,lcon)

[loss,loss_grad] = Bsfgrad(Ax,A,gamma,lcon);

if nargout>=2
   
    [grho_obj,grho_grad] = grhofun(x,rho,acon);  
    
else
   
     grho_obj = grhofun(x,rho,acon);
end

fobj = loss - lambda_rho*grho_obj;

if nargout>=2
    
    grad = loss_grad - lambda_rho*grho_grad;
end

