%basata sul video
clear;close all;clc;
%%
load("input_data.mat"); %year, snowshoe hare pelts, lynx pelts
inputdata = inputdata';

space = [1 2]; %lepri e linci

% t = inputdata(1,1:end);
t = 0:29;
pelts = inputdata(2:end,:);

%figure(1), waterfall(space,t',pelts');
%figure(2), plot(t,pelts(1,:)');


dt = t(2)-t(1);
X = pelts; param = 1;
X1 = X(:,1:end-param);
X2 = X(:,param+1:end);
u = X(:,1); %initial condition
[U,Sigma,V] = svd(X1,"econ");
S = U'*X2*V*diag(1./diag(Sigma)); %questa dovrebbe essere ATilde
[eV,D] = eig(S); %Eigenvectors and eigenvalues
mu = diag(D);
omega = log(mu)/(dt);
Phi = U*eV;
y0 = Phi\u; %pseudo-inverse initial conditions
u_modes = zeros(size(V,2),length(t));
for iter = 1:length(t)
    u_modes(:,iter) = y0.*exp(omega*t(iter));
end
u_dmd = Phi*u_modes;
%%
%figure(3), waterfall(space,t',abs(u_dmd.'));
t_plot=inputdata(1,1:end);
figure(4), hold on, set(gca,'Fontsize',20), grid on, plot(t_plot,1000*pelts(1,:)','b-o','LineWidth',2),plot(t_plot,1000*abs(u_dmd(1,:)'),"Color",[ 0.47 0.25 0.80],'LineWidth',2), xlabel("Year"), ylabel("Population")
title('Prey (Snowshore Hare)')
legend('Real Data','Exact DMD Reproduction'), hold off;
figure(5), hold on, set(gca,'Fontsize',20), grid on, plot(t_plot,1000*pelts(2,:)','r-o','LineWidth',2),plot(t_plot,1000*abs(u_dmd(2,:)'),"Color",[1.00 0.54 0.00],'LineWidth',2)
xlabel("Year"); ylabel("Population")
title('Predator (Lynx)')
legend('Real Data','Exact DMD Reproduction')
hold off;
% close all;
