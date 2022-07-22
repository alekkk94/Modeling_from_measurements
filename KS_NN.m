clear all; close all; clc;

% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - nu*u_xxxx,  periodic BCs

N = 300; %1024
x = 2*pi*(1:N)'/N;
u = -sin(x)+2*cos(2*x)+3*cos(3*x)-4*sin(4*x);
v = fft(u);
alpha = 87;
nu = 4/alpha;

% % % % % %
%Spatial grid and initial condition:
h = 0.01; %10^(-4);
k = [0:N/2-1 0 -N/2+1:-1]';
L = k.^2 - nu*k.^4; %% added the 4* for my specif
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 , 2) );
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
tmax = 140; nmax = round(tmax/h); nplt = floor((tmax/1000)/h); g = -0.5i*k;
tt = zeros(1,nmax);
uu = zeros(N,nmax);

for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2); 
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;     
    if mod(n,nplt)==0
        n;
        u = real(ifft(v));
        uu(:,n) = u; 
        tt(n) = t;
    end
end


%%
cutoff = tt > 120;
cutoff = cutoff & tt<130;
%  cutoff = tt > 0; cutoff = cutoff & tt<10;

figure(1)
contour(x/(2*pi),tt(cutoff),uu(:,cutoff).',[-10 -5 0 5 10]),shading interp, colormap("default")
% contour(x,tt,uu.'),shading interp, colormap(gray)

figure(2)
surf(x,tt(cutoff),uu(:,cutoff).'),shading interp, colormap("default"), view(2)
tsave = tt(cutoff);
xsave = x/(2*pi);
dt = h;
dx = 1/N;
usave = uu(:,cutoff).';

%save kuramoto_sivashinsky.mat xsave tsave usave dt dx
%% proviamo con DMD prima di fare tutto:
% input = uu(:,cutoff); input = input(:,1:end-1);
% output = uu(:,cutoff); output = output(:,2:end);
% 
% [U,S,V] = svd(input,'econ');
% cum_S = diag(S)/sum(sum(S));
% rango = 16;
% Ur=U(:,1:rango);
% Sr=S(1:rango,1:rango);
% Vr=V(:,1:rango);
% Atilde=Ur'*output*Vr/Sr;
% [W,Lambda] = eig(Atilde);
% 
% Phi = output*Vr*inv(Sr)*W; % Step 4
% Alpha1=Sr*Vr(1,:)';
% b_DMD = (W*Lambda)\Alpha1;
% 
% mu=diag(Lambda);
% omega=log(mu)/dt;
% 
% y0 = Phi\input(:,1);
% 
% t = 1:length(tt(cutoff));
% for iter = 1:length(tt(cutoff)) %teh DMD forecasts all the timesteps (form 1 to 200) 
%     u_modes(:,iter) =(y0.*exp(omega*(t(iter))));
% end
% 
% u_dmd= real(Phi*u_modes); 

%% train neural network
%input = uu(1:end-1,cutoff);
%output = uu(2:end,cutoff); %quelle vecchie, probabilmente sbagliate

input = uu(:,cutoff); input = input(:,1:end-1);
output = uu(:,cutoff); output = output(:,2:end);

net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'satlins';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'radbas';
% net.layers{3}.transferFcn = 'purelin';
net = train(net,input,output); %s x t
%% plotting

x0=usave(1,:).';
%x0=usave(1:end-1,1);
clear ynn
ynn(1,:)=x0;
for jj=2:length(tt(cutoff))
    y0=net(x0);
    ynn(jj,:)=y0.';
    %x0=[rho_try;y0(2:4)];
    x0=y0;
end
%%
figure(4)
contour(x/(2*pi),tt(cutoff),ynn(:,:),[-10 -5 0 5 10]),shading interp, colormap("default")

figure(5)
surf(x,tt(cutoff),ynn),shading interp, colormap("default"), view(2)
