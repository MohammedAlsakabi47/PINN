clc; clear all; close all
%% Basic waves:
fs = 3000;
f = 100;
t = 0:1/fs:3/f;

%%% circle equation
theta = 0:5:720;
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

%% Load Results:
load('results.mat');

%% Single Blade:
% figure
% grid on
freq_prog = []; % if the receiver is placed along x-axis
input_waves = [];
ground_truth_waves = [];
fourier_response = [];
result_progress = [];
ground_truth_progress = [];
% Create a figure with a specific size
figure('Position', [0, 100, 1400, 600]); % [left, bottom, width, height]
pause(1)
for ii=1:length(theta)
    subplot(2,3,1)
    cla; % Clears the current axes
    z = -1:0.01:1; % blade span
    scatter([z(1:end-1)*x_0(ii)], [z(1:end-1)*y_0(ii)], 'o', 'filled'); hold on
    scatter(z(end)*x_0(ii), z(end)*y_0(ii), 'o', 'filled')
    title('2 Blades Rotor - on x,y plane', 'FontSize', 14)
    xlabel('x', 'FontSize', 14)
    ylabel('y', 'FontSize', 14)
    legend('Blade Discrete Points','Tip point for Velocity Estimation')
    xlim([-2 2])
    ylim([-2 2])
    annotation('arrow', [0.179, 0.129], [0.75, 0.75], 'LineWidth', 2);

    % Add text above the arrow
    % The position is in normalized figure coordinates (0 to 1)
    text(-1.5, 0.1, 'Receiver', 'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 12);
    grid on

    subplot(2,3,2)
    scatter([x_0(ii) 0], [0, y_0(ii)], 'o', 'filled')
    xlim([-2 2])
    ylim([-2 2])
    title('Tip location on x and y axes', 'FontSize',14)
    xlabel('x', 'FontSize', 14)
    ylabel('y', 'FontSize', 14)
    grid on

    freq_prog = [freq_prog y_0(ii)*1000];
    
    received_wave = exp(j*2*pi*freq_prog(end)*t) + 0.3*randn(1,length(t));
    input_waves = [input_waves transpose(received_wave)];
    ground_truth_waves = [ground_truth_waves exp(j*2*pi*freq_prog(end)*t')];
    subplot(2,3,4)
    plot(real(received_wave))
    xlim([0 91])
    ylim([-2 2])
    title('Wave at receiver', 'FontSize',14)
    xlabel('Time Index', 'FontSize', 14)
    ylabel('Aplitude', 'FontSize', 14)
    grid on

    subplot(2,3,5)
    fourier = abs(fftshift(fft(received_wave)));
    % fourier_response = [fourier_response transpose(fourier)];
    plot(linspace(-1000,1000,91),abs(fourier))
    xlim([-1000 1000]);
    ylim([0 100]);
    title('Instantanous Spectrum of Wave at Receiver', 'FontSize',14)
    xlabel('Frequency Bin', 'FontSize', 14)
    ylabel('Power', 'FontSize', 14)
    grid on

    subplot(1,3,3)
    result_progress = [result_progress result(ii)];
    ground_truth_progress = [ground_truth_progress grount_truth(ii)];
    cla; % Clears the current axes
    plot(ground_truth_progress, 'LineWidth',2); hold on
    plot(result_progress, 'LineWidth',1)
    xlim([1 145])
    ylim([-110 110])
    title('Instantanous Linear Velocity - Ground Truth vs MLP Prediction', 'FontSize',14)
    xlabel('Time Index', 'FontSize', 14)
    ylabel('Velocity (m/s)', 'FontSize', 14)
    legend('GT', 'MLP')
    grid on
    pause(0.01)
end



