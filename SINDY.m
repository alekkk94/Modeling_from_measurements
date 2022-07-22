clear;close all;clc;
%% optdmd
addpath('./src');
addpath('./utils/');
load("input_data.mat"); %year, snowshoe hare pelts, lynx pelts
inputdata = inputdata';

space = [1 2]; %lepri e linci

% t = inputdata(1,1:end);

t_plot=inputdata(1,1:end);
dt = t_plot(2)-t_plot(1);
t = (t_plot-t_plot(1))/dt;
% t = t_plot;
pelts = inputdata(2:end,:);

[size_1,size_2] = size(pelts);

%figure(1), waterfall(space,t',pelts');
%figure(2), plot(t,pelts(1,:)');



X = pelts; param = 1;
shoe = X(1,:)'; %x
linx = X(2,:)';
dt=1;
for tt =1:size(X,2)-1
der_shoe_data(tt) = (X(1,tt+1)-X(1,tt))/dt; 
der_lynx_data(tt) = (X(2,tt+1)-X(2,tt))/dt; 
end
dx_data = [der_shoe_data.' der_lynx_data.'];
%% SINDY
polyorder = 2;  % search space up to fifth order polynomials
usesine = 0;    % no trig functions

n = 2; %order of the system

%u = reshape(X(:,2:end-1).',(size_2-2)*size_1,1);

Theta = poolData(X(:,1:end-1)',n,polyorder,usesine);
%Theta(:,3) = [];

%regression
%xi_inv=Theta\dx_data;
xi1_inv=Theta\der_shoe_data.';
xi2_inv=Theta\der_lynx_data.';

xi1_pinv=pinv(Theta)*der_shoe_data.';
xi2_pinv=pinv(Theta)*der_lynx_data.';


xi1_lasso=lasso(Theta,der_shoe_data,'Lambda',0.1);
xi2_lasso=lasso(Theta,der_lynx_data,'Lambda',0.1);

% xi1_lasso=lasso(Theta,der_shoe(1:end-1),'Lambda',0.1);
% xi2_lasso=lasso(Theta,der_lynx(1:end-1),'Lambda',0.1);

% Xi = [xi1_inv xi2_inv];
Xi = [xi1_lasso xi2_lasso];


% compute Sparse regression: sequential least squares
 lambda = 0.002;      % lambda is our sparsification knob.
 %Xi = sparsifyDynamics(Theta,dx_data,lambda,n);
%poolDataLIST({'Shoe','Lynx'},Xi,n,polyorder,usesine);

%ricostruzione approxximata
tspan = [0 30]; options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
t(1) = 0;
t=t'; x_piccolo = [shoe linx];
%% Bagging SINDY
num_cyclesS =  300;
tB=10;
%number of time points
nS = length(tB);
%number you want to choose
pxS = round(0.85*size(Theta,2)); 
%pxS = 3;
ensT = 0.65;
mOutBS = zeros(pxS,n,num_cyclesS);
libOutBS = zeros(pxS,num_cyclesS);

for jS = 1:num_cyclesS
     rs = RandStream('mlfg6331_64','Seed',jS);
     libOutBS(:,jS) = datasample(rs,1:size(Theta,2),pxS,'Replace',false)';
    mOutBS(:,:,jS) = sparsifyDynamics(Theta(:,libOutBS(:,jS)),dx_data,lambda,n);
end

inclProbBS = zeros(size(Theta,2),n);
for iii = 1:num_cyclesS
    for jjj = 1:n
        for kkk = 1:pxS
            if mOutBS(kkk,jjj,iii) ~= 0
                inclProbBS(libOutBS(kkk,iii),jjj) = inclProbBS(libOutBS(kkk,iii),jjj) + 1;
            end
        end
    end
end
inclProbBS = inclProbBS/num_cyclesS*size(Theta,2)/pxS;

XiD = zeros(size(Theta,2),n);
for iii = 1:n
    libEntry = inclProbBS(:,iii)>ensT;
%     XiBias = sparsifyDynamics(Theta(:,libEntry),dx_data(:,iii),lambda,1);
    XiBias=lasso(Theta(:,libEntry),dx_data(:,iii),'Lambda',0.1);
    XiD(libEntry,iii) = XiBias;
end