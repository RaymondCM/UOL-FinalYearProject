%% Enviroment
clc;
clear;

%% Read in DICOM file and Get heart rate

[file_name, root] = uigetfile('*.dcm');
file_path = strcat(root, file_name);
info = dicominfo(file_path);

ground_bpm = info.HeartRate;
capture_rate = 1000 / info.FrameTime;

%% Get File Path
[file_name,file_root] = uigetfile('*.txt');
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

max_x = period(find(y_power == max(y_power), 1, 'first'));

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

%% Estimate BPM over Samples
n_steps = 4;
x_step = floor(n / n_steps);

p = 1;
x = 0;

data  = 1:n_steps;

for x = 0 : n_steps - 1
    ps = 1 + (x_step * x);
    pe = ps + x_step;
    
    S = angles(ps:pe);
    SX = 1:size(S);

    SffY = fft(S); % Compute Fast Fourier Transform.
    SffY(1) = []; % Discard first value as it's the sum of all the frequencies.
    s_n = length(SffY);

    max_freq = 0.5; % Data is mirorred after this point so discard it.
    y_power = abs(SffY(1:floor(s_n * max_freq))) .^ 2; % Calc power of freq.

    s_freq = (1:s_n/2)/(s_n/2) * max_frequency; % Normalise X between 0 and 0.5
    s_period = 1./s_freq; % Will be the same as 1:size(Y)/2 * 2
    
    s_period(find(y_power == max(y_power), 1, 'first'))
    
    data(x+1) = s_period(find(y_power == max(y_power), 1, 'first'));
    data(x+1) = (60 * capture_rate) * (data(x + 1) / 1000);
end

Y1 = data;
x_bpm = (60 * capture_rate) * (max_x / 1000);

Y2 = x_bpm:x_bpm+length(Y1)-1;
Y2(Y2 > x_bpm) =  x_bpm;

Y3 = ground_bpm:ground_bpm+length(Y1)-1;
Y3(Y3 > ground_bpm) = ground_bpm;

%% Plot BPM over Time

figure('Name','Full Exhastive SAD Block Matching')
subplot(2, 1, 1);

plot(1:length(Y1), Y1, 1:length(Y2), Y2, 1:length(Y3), Y3)

legend('Estimated BPM of Split data', 'Overall BPM estimate', 'Ground Truth')
ylim([80 120]);
xticks(1:length(Y1))
xticklabels({'1:126','126:252','252:378','378:504'})
xlabel 'Data Split Range'
ylabel 'BPM'
title 'Estimated BPM vs Ground Truth'
