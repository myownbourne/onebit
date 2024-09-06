%% *********** The proximal mapping of the L1 function **************
%  filename: Prox_L1sphere  
%
%% ***************************************************************
%  This code aims to compute the solution to the nonconvex problem  
%  
%   min 0.5 ||x-g||^2 + lambda||x||_1 s.t. ||x||=1
%    
%% ****************************************************************
  
%% ***************************************************************
%% Copyright by Chen Kai and Shaohua Pan, 2021/12/1

%% ***************************************************************

function xp = Prox_L1sphere(g,lambda)

p = size(g,1);

xsol = zeros(p,1);

temp_xp = zeros(p,1);

[sg,idx] = sort(abs(g),'descend');

sign_g = sign(g);

if lambda >= sg(1)
    
    xsol(1) = 1;
    
    temp_xp(idx) = xsol;
    
    xp = temp_xp.*sign_g;

elseif lambda<= sg(end)
    
    sg_lambda = sg-lambda;
    
    xsol = sg_lambda/norm(sg_lambda);
    
    temp_xp(idx) = xsol;
    
    xp = temp_xp.*sign_g;

else
    
    for k = 1:p-1
        
        if lambda>sg(k+1)&&lambda<sg(k)
            
            sg_lambda = sg(1:k)-lambda;
            
            temp_vec = sg_lambda/norm(sg_lambda);
            
            xsol(1:k) = temp_vec;
           
            temp_xp(idx) = xsol;
            
            xp = temp_xp.*sign_g;
            
            return;           
        
        end          
        
    end   
    
end


