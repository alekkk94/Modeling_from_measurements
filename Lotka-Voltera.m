clear all, close all, clc
load pelts.mat
xy_data =  data(1:2, :)';
xydot_data = diff(xy_data);
xy_data = xy_data(2:end,:);

% initial guess
params = [  1;    % b
            1;  % p
            1;  % r
            1  ]'; % d   

params = lsqcurvefit(@LotkaVolterra,params,xy_data,xydot_data)

plot(xy_data(:,1),'b')
hold on
plot(xy_data(:,2), 'r')
hold on

xy_LV = LotkaVolterra(params,xy_data);

plot(xy_LV(:,1), 'b--')
hold on
plot(xy_LV(:,2), 'r--')

legend('Data X','Data Y', 'LV X', 'LV Y')
title('Data and Lotka Volterra Best Fit Model')

mae = mean(abs(xy_data - xy_LV))


% 0.5428    0.0068    0.0078    0.4387
function xydot = LotkaVolterra(params, xy_data)
b = params(1);
p = params(2);
r = params(3);
d = params(4);

x = xy_data(:,1);
y = xy_data(:,2);

xdot = b*x - p*y.*x;
ydot = r*x.*y - d*y;
xydot = [xdot, ydot];
end
