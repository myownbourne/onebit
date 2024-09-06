%% ****************************************************************
% filename: BsAPG_scad
%% ****************************************************************
% Kai Chen and Shaohua Pan
% gamma: Moreau envelope parameter
% lamda,rho,a: SCAD parameters

function [xopt,fopt] = BsAPG_scad(A,x,OPTIONS,lambda,sigma)

 if isfield(OPTIONS,'maxiter');          maxiter    = OPTIONS.maxiter;    end
 if isfield(OPTIONS,'printyes');         printyes   = OPTIONS.printyes;   end
 if isfield(OPTIONS,'Asnorm');           Asnorm     = OPTIONS.Asnorm;     end
 if isfield(OPTIONS,'tol');              tol        = OPTIONS.tol;        end

acon = 5.0;

acon1 = acon+1;

gamma = 0.05;

rho = 10;

Lip = (1/gamma)*Asnorm+lambda*rho^2*max(acon1/2,acon1/(2*(acon-1)));

tau = 1/Lip; 

lambda_rho = lambda*rho; 

lambda_tau = lambda_rho*tau;

%%
%% ***********************  Main Loop ***************************

tstart = clock;

diffobj_list = zeros(1,maxiter);

xold = x;  

Ax = A*x;  Axold = Ax;

fobj = Bsfgrad_scad(x,Ax,A,gamma,rho,acon,lambda_rho,sigma);

objold = fobj + lambda_rho*sum(abs(xold));

told = 1;  t = 1;

for iter = 1:maxiter
    
    beta = min((told-1)/t,0.235);
    
    beta1 = 1+beta;
    
    xt = beta1*x - beta*xold;           % extrapolation step
    
  %% ******************** to compute xnew *************************
   
    Axt = beta1*Ax - beta*Axold;
    
    [fobjxt,gradxt] = Bsfgrad_scad(xt,Axt,A,gamma,rho,acon,lambda_rho,sigma);

    tempxt = xt - tau*gradxt;
 
    xnew = Prox_L1sphere(tempxt,lambda_tau);
    
    Axnew = A*xnew;
    
    fobj = Bsfgrad_scad(xnew,Axnew,A,gamma,rho,acon,lambda_rho,sigma);
    
    obj = fobj + lambda_rho*sum(abs(xnew));
    
    %% ************** generate the new iterate xnew ***************
    
    ttime = etime(clock,tstart);
    
    opt_measure = norm(xnew-xt); 

    diff_obj = (obj-objold)/max(1,abs(obj));
    
    diffobj_list(iter) = abs(diff_obj);
    
    if (printyes)&&(mod(iter,10)==0)
        
        fprintf('\n %3d    %3.2e     %3.2e     %3.2e      %3.1f    %3.1f    %3.2d',iter,opt_measure,diff_obj,tau,beta,ttime,nnz(x));
        
    end
    
    %%
    %% ************* check stopping criterion ******************
    %% 
    if (iter>=100)
        
        if (opt_measure<tol)||(iter==maxiter)||(max(diffobj_list(iter-9:iter))<=1e-10)
            
            xopt = xt;  fopt = fobjxt;
            
            return;
        end
    end
    
    xold = x;  x=xnew;   
    
    Axold = Ax; Ax = Axnew;
    
    objold = obj;  
    
    told = t;
    
    t = 0.5*(1+sqrt(1+4*told^2));
end

xopt = xt;  fopt = fobjxt;


 
    
