%% *********** the proximal map of the L0 function *****************
%  filename: Prox_L0sphere  
%
%% ***************************************************************
%  This code aims to compute the solution to the nonconvex problem  
%  
%   min 0.5 ||x - g||^2 + lambda||x||_0£ºs.t. ||x||=1
%    
%% ****************************************************************
  
%% ***************************************************************
%% Copyright by Shaohua Pan and Wuyu Qia, 2018/11/8
%  our paper: "A globally and linearly convergent PGM for zero-norm 
%  regularized quadratic optimization with sphere constraint"
%% ***************************************************************

function xp = Prox_L0sphere(g,lambda)

p = size(g,1);

xsol = zeros(p,1);

temp_xp = zeros(p,1);

[sg,idx] = sort(abs(g),'descend');

sign_g = sign(g);

norm_sg = norm(sg);

if lambda >= sg(1)
    
    xsol(1) = 1;
    
    temp_xp(idx) = xsol;
    
    xp = temp_xp.*sign_g;

elseif lambda<=norm_sg-norm(sg(1:p-1))
    
    xsol = sg/norm_sg;
    
    temp_xp(idx) = xsol;
    
    xp = temp_xp.*sign_g;

else
    
    for k = 2:p-1
         
        hk = norm(sg(1:k)) - norm(sg(1:k-1));
        
        if lambda>=hk 
            
            temp_vec = sg(1:k-1)/norm(sg(1:k-1));
            
            xsol(1:k-1) = temp_vec;
           
            temp_xp(idx) = xsol;
            
            xp = temp_xp.*sign_g;
            
            return;          
        
        end          
        
    end   
    
end


