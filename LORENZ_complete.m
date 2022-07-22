clear; close all; clc;
%%
% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10;
%rho_test=[10 28 35];
rho_test=[10 28 35];
%rho_test = 10:2:40;

ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);


input=[]; output=[];

for k = 1:50 % training trajectories
    for j=1:numel(rho_test)

        rho=rho_test(j);

        Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
            rho * x(1)-x(1) * x(3) - x(2) ; ...
            x(1) * x(2) - b*x(3)         ]);

        x0=30*(rand(3,1)-0.5);
        %x0=30*ones(3,1);

        [t,y] = ode45(Lorenz,t,x0);
       % [t,y] = ode45(@loretz_eq,t,x0,[],sig,rho,b);
        input=[input; y(1:end-1,:)];
        output=[output; y(2:end,:)];
        plot3(y(:,1),y(:,2),y(:,3)), hold on
        plot3(x0(1),x0(2),x0(3),'ro')

    end
end

grid on, view(-23,18)

% for j=1:1 % training trajectories
%     x0=30*ones(3,1);
%     rho=rho_test(j);
%     [t,y] = ode45(@loretz_eq,t,x0,[],sig,rho,b);
%     input=[input; y(1:end-1,:)];
%     output=[output; y(2:end,:)];
%     plot3(y(:,1),y(:,2),y(:,3)), hold on
%     plot3(x0(1),x0(2),x0(3),'ro')
% end


rho_tested=[];
for l=1:numel(rho_test)
    %rho_tested=[rho_tested;rho_test(l)*ones(numel(t)-1,1)]; %old
    rho_tested=[rho_tested;rho_test(l)*ones(k*(numel(t)-1),1)];
end
input_NN(:,1)=rho_tested;
input_NN(:,2:4)=input;
output_NN(:,1)=rho_tested;
output_NN(:,2:4)=output;



%% train the Neural Network
net = feedforwardnet([10 10 10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input_NN.',output_NN.');
%%

set(gcf,'color','w');
rho_try=40;
% x0=30*(ones(3,1));

x0=20*(rand(3,1)-0.5);

%figure(22)
%dt=0.001; T=8; t=0:dt:T;
%[t,y] = ode45(@loretz_eq,t,x0,[],sig,rho_try,b);
Lorenz1 = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  rho_try * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]); 

%[t,y] = ode45(@loretz_eq,t,x0,[],sig,rho_try,b);
 [t,y_L] = ode45(Lorenz1,t,x0);

grid on
%rho is used as a input
x0=[rho_try;x0];
clear ynn
ynn(1,:)=x0;
for jj=2:length(t)
    y0=net(x0);
    ynn(jj,:)=y0.';
    x0=[rho_try;y0(2:4)];
    %x0=y0;
end
figure(2)
plot3(y_L(:,1),y_L(:,2),y_L(:,3)), hold on
plot3(ynn(:,2),ynn(:,3),ynn(:,4),':','Linewidth',[2])
legend('ODE45','NN')
xlabel('x');
ylabel('y');
zlabel('z');

figure(3)
subplot(3,1,1), plot(t,y_L(:,1),t,ynn(:,2),'Linewidth',[2]),legend('ODE45','NN'),xlabel('Time'); ylabel('x');
grid on
subplot(3,1,2), plot(t,y_L(:,2),t,ynn(:,3),'Linewidth',[2]),legend('ODE45','NN'),xlabel('Time'); ylabel('y');
grid on
subplot(3,1,3), plot(t,y_L(:,3),t,ynn(:,4),'Linewidth',[2]),legend('ODE45','NN'),xlabel('Time'); ylabel('z');
grid on
