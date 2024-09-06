function [X,yf,ytrue,xopt,T,norm_x] = gen1bcs_real_data(Ctype,m,n,cfac,nf,r)
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

[xopt,T,norm_x] = sparse(n);
ytrue    = sign(X(:,T)*xopt(T));
ynoise   = sign(X(:,T)*xopt(T)+nf*randn(m,1));
yf       = flip(ynoise,r,m);
end

% generate a sparse vector ------------------------------------------------
function [x,T,norm_x] = sparse(n)
data = load('signal.txt');
signal = data(1:n);
xtrue_b = dct(signal);
norm_x = norm(xtrue_b);
x = xtrue_b/norm_x;
T = find(x); 
end

% flip the signs of a vector ----------------------------------------------
function yf   = flip(ynoise,r,m)
yf   = ynoise;
Rs    = randperm(m);
T    = Rs(1:ceil(r*m));
yf(T)= -yf(T);
end
