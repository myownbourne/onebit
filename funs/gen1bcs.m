function [X,yf,ytrue,xopt,T] = gen1bcs(Ctype,xtype,m,n,k,cfac,nf,r)
% This file aims at generating data of 2 types of Gaussian samples
% for 1-bit compressed sensing
% Inputs:
%       Ctype   -- the type of sample matrix: 'Ind'or 'Cor'
%       xtype   -- the type of true signal: 'Normal'or 'Bernou'
%       m       -- number of samples
%       n       -- number of features
%       k       -- sparsity of the true singnal
%       nf      -- nosie factor
%       r       -- flipping ratio belonging to (0,1)
%      cfac     -- correlation factor belonging to (0,1)
% Outputs:
%       X       --  samples data, m-by-n matrix
%       xopt    --  n-by-1 vector, i.e., the true signal
%       ytrue   --  m-by-1 vector, i.e., sign(X*xopt)
%       yf      --  m-by-1 vector, ytrue after flapping some signs
%
% written by Shenglong Zhou, 19/07/2020
switch Ctype
    case 'Ind'
        X = randn(m,n);
    case 'Cor'
        S = cfac.^(abs((1:n)-(1:n)'));
        X = mvnrnd(zeros(n,1),S,m);
end

[xopt,T] = sparse(n,k,xtype);
ytrue    = sign(X(:,T)*xopt(T));
ynoise   = sign(X(:,T)*xopt(T)+nf*randn(m,1));
yf       = flip(ynoise,r,m);
end

% generate a sparse vector ------------------------------------------------
function [x,T] = sparse(n,k,xtype)
x    = zeros(n,1);
Rs   = randperm(n);
T    = Rs(1:k);
x(T) = randn(k,1);
switch xtype
    case 'Normal'
        x(T) = x(T)/norm(x(T));
end
end

% flip the signs of a vector ----------------------------------------------
function yf   = flip(ynoise,r,m)
yf   = ynoise;
Rs    = randperm(m);
T    = Rs(1:ceil(r*m));
yf(T)= -yf(T);
end
