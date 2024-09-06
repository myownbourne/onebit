seednum = 1*2022;
rand('seed',seednum);  
randn('seed', seednum);

m = 5000;
n = 500;
rflip = 0.1; %or 0.1   
omega = 0.3;     
cfac = 0.3;
nflip = ceil(rflip*m);

data = load('signal.txt');
signal = data(1:n);
[Phi,yfn,ytrue,xtrue,supp,norm_x] = gen1bcs_real_data('Cor',m,n,cfac,omega,rflip);

%Psi = dctmtx(N);
%x = Psi*signal;
%s_true = idct(xtrue);

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
    
OPTIONS_APG.Asnorm = Asnorm;
OPTIONS_APG.tol = 1.0e-6;
OPTIONS_APG.printyes = 0;
OPTIONS_APG.maxiter = 5000;

K = 150;

x1 = PIHT(yfn,Phi,K,-0.2,1,1);
x1_a = x1*norm_x;
s1 = idct(x1_a);

x2 = BIHT_AOP_flip(yfn, Phi, Phi'*yfn,K,nflip,1,100,1);
x2 = x2/norm(x2);
x2_a = x2*norm_x;
s2 = idct(x2_a);

x3 = PIHT_AOP_flip(yfn,Phi,Phi'*yfn,K,nflip,1,100,1,0.05);
x3 = x3/norm(x3);
x3_a = x3*norm_x;
s3 = idct(x3_a);

out4 = GPSP_new(Phi,yfn,K,nflip);
x4 = out4.x;
x4_a = x4*norm_x;
s4 = idct(x4_a);

tempx0 = sum(A,1)';
x0 = tempx0/norm(tempx0);
x5 = BsAPG_scad(A,x0,OPTIONS_APG,2,0.8); 
x5_a = x5*norm_x;
s5 = idct(x5_a);

x52 = BsAPG_scad(A,x0,OPTIONS_APG,4,0.8); 
x52_a = x52*norm_x;
s52 = idct(x52_a);

x6 = BsAPG_L0(A,x0,OPTIONS_APG,80,0.8);
x6_a = x6*norm_x;
s6 = idct(x6_a);

x62 = BsAPG_L0(A,x0,OPTIONS_APG,160,0.8);
x62_a = x62*norm_x;
s62 = idct(x62_a);

x7 = pdasc(Phi,Phi',yfn);
esupp = find(x7);
Phiesupp = Phi(:,esupp);
x7(esupp) = (Phiesupp'*Phiesupp)\(Phiesupp'*yfn);
x7 = x7/norm(x7);
x7_a = x7*norm_x;
s7 = idct(x7_a);

[x8,lam,ithist] = wpdasc(Phi,Phi',yfn);
esupp = find(x8);
Phiesupp = Phi(:,esupp);
x8(esupp) = (Phiesupp'*Phiesupp)\(Phiesupp'*yfn);
x8 = x8/norm(x8);
x8_a = x8*norm_x;
s8 = idct(x8_a);

err1 = norm(s1-signal)/norm(signal);
err2 = norm(s2-signal)/norm(signal);
err3 = norm(s3-signal)/norm(signal);
err4 = norm(s4-signal)/norm(signal);
err5 = norm(s5-signal)/norm(signal);
err52 = norm(s52-signal)/norm(signal);
err6 = norm(s6-signal)/norm(signal);
err62 = norm(s62-signal)/norm(signal);
err7 = norm(s7-signal)/norm(signal);
err8 = norm(s8-signal)/norm(signal);

fs = 8;


fig = figure;
set(fig, 'Position', [0, 0, 1000, 800]);

subplot(5,2,1);
plot(signal);hold on
plot(s1)
lgd = legend({'Signal', 'PIHT'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err1))])

subplot(5,2,2);
plot(signal);hold on
plot(s2)
lgd = legend({'Signal', 'BIHT-AOP'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err2))])

subplot(5,2,3);
plot(signal);hold on
plot(s3)
lgd = legend({'Signal', 'PIHT-AOP'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err3))])

subplot(5,2,4);
plot(signal);hold on
plot(s4)
lgd = legend({'Signal', 'GPSP'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err4))])

subplot(5,2,5);
plot(signal);hold on
plot(s5)
lgd = legend({'Signal', 'PGe-scad$(\lambda=2)$'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err5))])

subplot(5,2,6);
plot(signal);hold on
plot(s52)
lgd = legend({'Signal', 'PGe-scad$(\lambda=4)$'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err52))])

subplot(5,2,7);
plot(signal);hold on
plot(s6)
lgd = legend({'Signal', 'PGe-znorm$(\lambda=80)$'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err6))])

subplot(5,2,8);
plot(signal);hold on
plot(s62)
lgd = legend({'Signal', 'PGe-znorm$(\lambda=160)$'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err62))])

subplot(5,2,9);
plot(signal);hold on
plot(s7)
lgd = legend({'Signal', 'PDASC'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err7))])

subplot(5,2,10);
plot(signal);hold on
plot(s8)
lgd = legend({'Signal', 'WPDASC'}, 'Interpreter', 'latex','Location', 'north'); lgd.FontSize = fs;
title(['err ' num2str(sprintf('%.4f', err8))])

%exportgraphics(gcf,'fig_result_001.eps')

[err1, err2, err3, err4,err5,err52,err6,err62,err7,err8]







