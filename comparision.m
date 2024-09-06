n = 2000;        % Signal dimension
m = 800;         % Number of measurements
K = 10;           % signal sparsity
rflip = 0.15;      % probability of sign flips
omega = 0.1;      % noise level 
cfac = 0.1;       % coorelation
nflip = ceil(rflip*m);

ns = 8;
ntest = 50;
total_err = zeros(ns,1);
total_time = zeros(ns,1);
total_acc = zeros(ns,1);
total_FNR = zeros(ns,1);
total_FPR = zeros(ns,1);

OPTIONS_APG.tol = 1.0e-6;

OPTIONS_APG.printyes = 0;

OPTIONS_APG.maxiter = 2000;

for ii = 1:ntest
    
    XX = sprintf('\n the %dth test problem',ii);
    disp(XX)
    
    seednum = ii*2022;
    rand('seed',seednum);   % fix seed
    randn('seed', seednum); % fix seed
    
    %% ********************  Generate data ************************
    
    [Phi,yfn,ytrue,xtrue,supp] = gen1bcs('Cor','Normal',m,n,K,cfac,omega,rflip);
    
    A = yfn.*Phi;
    
    %%
    %% estimate Lipschitz constant = largest singular value of A.
    %%
    AAT_is_identity = 0;
    Ax = randn(m,1);
    Ay = A*(A'*Ax);
    if (norm(Ax-Ay) < 1e-16*norm(Ax))
        AAT_is_identity = 1;
        Asnorm = 1;
    end
    if (AAT_is_identity==0)
        options.tol   = 1e-6;
        options.issym = true;
        options.disp  = 0;
        options.v0    = randn(m,1);
        Asnorm = eigs(@(y)A*(A'*y),m,1,'LM',options);
    end
    
    OPTIONS_APG.Asnorm = sqrt(Asnorm);
    
    %% ***************** PIHT *************************************
    fprintf('------ PIHT  ----------\n')
    
    tstart = clock;
    
    x1 = PIHT(yfn,Phi,K,-0.2,1,1);
    
    ttime1 = etime(clock,tstart);
    
    total_time(1) = total_time(1) + ttime1;
    
    total_err(1) = total_err(1) + norm(xtrue-x1);
    
    total_acc(1) = total_acc(1) + nnz(sign(Phi*x1)-ytrue)/m;

    absx1 = abs(x1);
    
    esupp = find(absx1>=1.0e-5*max(absx1));
    
    FNR = length(setdiff(supp,esupp))/K;
    
    FPR = length(setdiff(esupp,supp))/(n-K);
    
    total_FNR(1) = total_FNR(1) + FNR;
    
    total_FPR(1) = total_FPR(1) + FPR;
    
    %% ***************** BIHT-AOP *********************************
    
    fprintf('------ BIHT-AOP  ------\n')
    alpha   = 1;
    tstart = clock;
    x2 = BIHT_AOP_flip(yfn, Phi, Phi'*yfn,K,nflip,1,100,alpha);
    x2 = x2/norm(x2);
    ttime2 = etime(clock,tstart);
    total_time(2) = total_time(2) +  ttime2;
    total_err(2) = total_err(2) + norm(xtrue-x2);
    total_acc(2) = total_acc(2)+nnz(sign(Phi*x2)-ytrue)/m;
    
    absx2 = abs(x2);
    
    esupp = find(absx2>=1.0e-5*max(absx2));
    
    FNR = length(setdiff(supp,esupp))/K;
    
    FPR = length(setdiff(esupp,supp))/(n-K);
    
    total_FNR(2) = total_FNR(2) + FNR;
    
    total_FPR(2) = total_FPR(2) + FPR;
    
    %% ***************** PIHT-AOP *********************************
    
    fprintf('------ PBAOP  ------\n')
    alpha   = 1;
    tstart = clock;
    x3 = PIHT_AOP_flip(yfn,Phi,Phi'*yfn,K,nflip,1,100,alpha,0.05);
    x3 = x3/norm(x3);
    ttime3 = etime(clock,tstart);
    total_time(3) = total_time(3) + ttime3;
    total_err(3) = total_err(3) + norm(xtrue-x3);
    total_acc(3) = total_acc(3) + nnz(sign(Phi*x3)-ytrue)/m;
    
    absx3 = abs(x3);
    
    esupp = find(absx3>=1.0e-5*max(absx3));
    
    FNR = length(setdiff(supp,esupp))/K;
    
    FPR = length(setdiff(esupp,supp))/(n-K);
    
    total_FNR(3) = total_FNR(3) + FNR;
    
    total_FPR(3) = total_FPR(3) + FPR;
    
    %% ********************** GPSP *********************************
    
    fprintf('------ GPSP ------\n')
    tstart = clock;
    out4 = GPSP(Phi,yfn,K,nflip);
    x4 = out4.x;
    ttime4 = etime(clock,tstart);
    total_time(4) = total_time(4) + ttime4;
    total_err(4) = total_err(4) + norm(xtrue-x4);
    total_acc(4) = total_acc(4)+nnz(sign(Phi*x4)-ytrue)/m;
    
    absx4 = abs(x4);
    
    esupp = find(absx4>=1.0e-5*max(absx4));
    
    FNR = length(setdiff(supp,esupp))/K;
    
    FPR = length(setdiff(esupp,supp))/(n-K);
    
    total_FNR(4) = total_FNR(4) + FNR;
    
    total_FPR(4) = total_FPR(4) + FPR;
    
    %% ********************** PGe-Scad ****************************
    fprintf('------ SCAD ------\n')
    tempx0 = sum(A,1)';
    x0 = tempx0/norm(tempx0);
    tstart = clock;
    x5 = BsAPG_scad(A,x0,OPTIONS_APG,4,0.8); 
    ttime5 = etime(clock,tstart);
    total_time(5) = total_time(5) + ttime5;
    total_err(5) = total_err(5) + norm(xtrue-x5);
    total_acc(5) = total_acc(5)+nnz(sign(Phi*x5)-ytrue)/m;
    
    absx5 = abs(x5);
    
    esupp = find(absx5>=1.0e-5*max(absx5));
    
    FNR = length(setdiff(supp,esupp))/K;
    
    FPR = length(setdiff(esupp,supp))/(n-K);
    
    total_FNR(5) = total_FNR(5) + FNR;
    
    total_FPR(5) = total_FPR(5) + FPR;
    
  %% ********************** PGe-Znorm ****************************
    fprintf('------ L0 ------\n')
    tstart = clock;
    x6 = BsAPG_L0(A,x0,OPTIONS_APG,8,0.8);
    ttime6 = etime(clock,tstart);
    total_time(6) = total_time(6) + ttime6;
    total_err(6) = total_err(6) + norm(xtrue-x6);
    total_acc(6) = total_acc(6)+nnz(sign(Phi*x6)-ytrue)/m;
    absx6 = abs(x6);
    
    esupp = find(absx6>=1.0e-5*max(absx6));
    
    FNR = length(setdiff(supp,esupp))/K;
    
    FPR = length(setdiff(esupp,supp))/(n-K);
    
    total_FNR(6) = total_FNR(6) + FNR;
    
    total_FPR(6) = total_FPR(6) + FPR;
    
    %% ************************* PDASC ****************************
    
    fprintf('------L1-LS(PDASC) ------\n')
    tstart = clock;
    x7 = pdasc(Phi,Phi',yfn);
    ttime7 = etime(clock,tstart);
    esupp = find(x7);
    Phiesupp = Phi(:,esupp);
    x7(esupp) = (Phiesupp'*Phiesupp)\(Phiesupp'*yfn);
    x7 = x7/norm(x7);
    
    total_time(7) = total_time(7) + ttime7;
    total_err(7) = total_err(7) + norm(xtrue-x7);
    total_acc(7)=total_acc(7)+nnz(sign(Phi*x7)-ytrue)/m;
    
    absx7 = abs(x7);
    
    esupp = find(absx7>=1.0e-5*max(absx7));
    
    FNR = length(setdiff(supp,esupp))/K;
    
    FPR = length(setdiff(esupp,supp))/(n-K);
    
    total_FNR(7) = total_FNR(7) + FNR;
    
    total_FPR(7) = total_FPR(7) + FPR;
    
    %% ************************* PDASC ****************************
    
    fprintf('------ LQ-LS(WPDASC) ------\n')
    tstart = clock;
    [x8,lam,ithist] = wpdasc(Phi,Phi',yfn);
    ttime8 = etime(clock,tstart);
    esupp = find(x8);
    Phiesupp = Phi(:,esupp);
    x8(esupp) = (Phiesupp'*Phiesupp)\(Phiesupp'*yfn);
    x8 = x8/norm(x8);
    total_time(8) = total_time(8) + ttime8;
    total_err(8) = total_err(8) + norm(xtrue-x8);
    total_acc(8) = total_acc(8) + nnz(sign(Phi*x8)-ytrue)/m;
    
    absx8 = abs(x8);
    
    esupp = find(absx8>=1.0e-5*max(absx8));
    
    FNR = length(setdiff(supp,esupp))/K;
    
    FPR = length(setdiff(esupp,supp))/(n-K);
    
    total_FNR(8) = total_FNR(8) + FNR;
    
    total_FPR(8) = total_FPR(8) + FPR;
end
% show results
fprintf('------ average  error ----------\n')
total_err'/ntest
fprintf('------ acc ----------\n')
total_acc'/ntest
fprintf('------  FNR ----------\n')
total_FNR'/ntest
fprintf('------  FPR ----------\n')
total_FPR'/ntest
fprintf('------ average   time ----------\n')
total_time'/ntest
