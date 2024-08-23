clc; clear all; close all
%% Parameters:
f = 100;
n = 0:0.02:1;
wave = exp(-j*2*pi*f*n);
plot(real(wave));

%%

%%% circle equation
omega = 1; %in rps
fs = 1;
theta = 0:1/fs:360;
theta_in_rad = theta*pi/180;
r = 1;

x_0 = r*cosd(theta);
y_0 = r*sind(theta);

x_72 = r*cosd(theta-72);
y_72 = r*sind(theta-72);

x_90 = r*cosd(theta-90);
y_90 = r*sind(theta-90);

x_120= r*cosd(theta-120);
y_120 = r*sind(theta-120);

x_144 = r*cosd(theta-144);
y_144 = r*sind(theta-144);

x_180 = r*cosd(theta-180);
y_180 = r*sind(theta-180);

x_216 = r*cosd(theta-216);
y_216 = r*sind(theta-216);

x_240 = r*cosd(theta-240);
y_240 = r*sind(theta-240);

x_270 = r*cosd(theta-270);
y_270 = r*sind(theta-270);

x_288 = r*cosd(theta-288);
y_288 = r*sind(theta-288);

figure(1)
grid on
for ii=1:length(theta)
    % tic
    z = 1; % blade span
    subplot(2,2,1)
    scatter([z*x_0(ii) z*x_180(ii)], [z*y_0(ii) z*y_180(ii)], 'o', 'filled');
    title('2 Blades', 'FontSize', 20)
    xlim([-2 2])
    ylim([-2 2])
    grid on

    subplot(2,2,2)
    scatter([z*x_0(ii) z*x_120(ii) z*x_240(ii)], [z*y_0(ii) z*y_120(ii) z*y_240(ii)], 'o', 'filled');
    title('3 Blades', 'FontSize', 20)
    xlim([-2 2])
    ylim([-2 2])
    grid on
    
    subplot(2,2,3)
    scatter([x_0(ii)*z x_90(ii)*z x_180(ii)*z x_270(ii)*z], [y_0(ii)*z y_90(ii)*z y_180(ii)*z y_270(ii)*z], 'o', 'filled');
    title('4 Blades', 'FontSize', 20)
    xlim([-2 2])
    ylim([-2 2])
    grid on

    subplot(2,2,4)
    scatter([x_0(ii)*z x_72(ii)*z x_144(ii)*z x_216(ii)*z x_288(ii)*z], [y_0(ii)*z y_72(ii)*z y_144(ii)*z y_216(ii)*z y_288(ii)*z], 'o', 'filled');
    title('5 Blades', 'FontSize', 20)
    xlim([-2 2])
    ylim([-2 2])
    grid on
    % toc

    pause(0.01)

    % 


    % Calculating linear velocity directly (using y-axis)
    subplot(2,3,5)
    plot(ones(1,22))
    


end


