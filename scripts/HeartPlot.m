%% Enviroment
clc;
clear; 

%% Get File Path
[file_name,file_root] = uigetfile('*.txt','Select the BlockMatching raw file');
data_path = strcat(file_root,file_name);

%% Read and parse file
data = tdfread(data_path, '\t');
angles = data.Angle_00x2D360;
magnitude = data.Magnitude;

weighted_angles = angles .* magnitude;

%% Original Plot
Y = angles(1:end);
X = 1:size(Y);

%% Calculate Frequency and Period
ffY = fft(Y); % Compute Fast Fourier Transform.
ffY(1) = []; % Discard first value as it's the sum of all the frequencies.
n = length(ffY);

max_frequency = 0.5; % Data is mirorred after this point so discard it.
y_power = abs(ffY(1:floor(n * max_frequency))) .^ 2; % Calc power of freq.

freq = (1:n/2)/(n/2) * max_frequency; % Normalise X between 0 and 0.5
period = 1./freq; % Will be the same as 1:size(Y)/2 * 2

%% Plot FFT and Original 
figure('Name','Full Exhastive SAD Block Matching')
subplot(2, 1, 1);

plot(X, Y)
xlim([0 n])
xlabel 'Frame Number'
ylabel 'Angular Motion'
title 'Average Angular Motion of Apical Four Chamber View'

subplot(2, 1, 2);
plot(period, y_power);
xlim([0 n])
xlabel 'Heart Beats over Frames (Freq Hz)'
ylabel 'Frequency Power (Amplitude)'
title 'Fast Fourier Transform Frequencies'