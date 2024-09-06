function x = PIHT(y, Phi, K, tau, bias, alpha)
%%
% PIHT is the function used to recover signals from 1-bit measurements in 
% 1-bit compressive sensing framework. It uses the iterative hard
% thresholding to minimize the sum of the pinball loss, with the unit
% sphere constraint and the sparsity constraint.
%
% Inputs are 
%            y:   The 1-bit measurements 
%          Phi:   The measurement system (M-by-N matrix)
%            K:   sparsity of the signal (only K components of x are
%                 non-zero)
%          tau:   The slope of the pinball loss for the negative side
%         bias:   The bias in the pinball loss (default value: 1)  
%        alpha:   The step-length (default value: 1)                     
%
% Outputs are
%            x:   The recovered signal
%                                                                       
%
% Authors:    Xiaolin Huang (xiaolin.huang@esat.kuleuven.be), Lei Shi(leishi@fudan.edu.cn),
%             Ming Yan (basca.yan@gmail.com), Johan A.K. Suykens(johan.suykens@esat.kuleuven.be)
%    Date:    2015-4-26(KU Leuven)
%
% Reference:  X. Huang, L. Shi, M. Yan, J.A.K. Suykens, Pinball Loss Minimization 
%      for One-bit Compressive Sensing. Internal Report 15-76, ESAT-SISTA, KU Leuven.
%
%%


if nargin == 4
    bias = 1;            % default bias term 
end

if nargin == 5
    alpha = 1;           % default step length
end

[~, N] = size(Phi);

% Negative function [.]_-
neg = @(in) in.*(in <0);

A = @(in) sign(Phi*in);


maxiter = 100;          % maximum iteration
htol = 0;                % acceptable number of wrong signs

x = zeros(N,1);          % initial guess
hd = Inf;

ii=0;                    % iteration count
while(htol < hd)&&(ii < maxiter)
	% Get gradient
    f_value = y.*(Phi*x);                 % analog value
    gradient = -2*y;                      % gradient for f_valuev < bias
    indx_p = find(f_value > bias);
    gradient(indx_p) = 2*tau*y(indx_p);   % gradient for strongly correctly classified points
  
    g = Phi'*gradient;
    
	% Step
	a = x - alpha*g;

	% Best K-term (threshold)
	[trash, aidx] = sort(abs(a), 'descend');
	a(aidx(K+1:end)) = 0;

	% Update x
	x = a;

	% Measure hammning distance to original 1bit measurements
	hd = nnz(y - A(x));
	ii = ii+1;
end


% Now project to sphere
x = x/norm(x);











