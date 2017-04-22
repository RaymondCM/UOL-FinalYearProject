% Get File Path
[in_file_name,in_file_root] = uigetfile('*.dcm','Select the DICOM image file');
in_data_path = strcat(in_file_root, in_file_name);

% Read in DICOM file
D = dicomread(in_data_path);

% Create VideoWriter Object
[out_file_name, out_file_root] = uiputfile('*.mp4', 'Select MPEG-4 Part 14 save location');
out_data_path = strcat(out_file_root, out_file_name);
V = VideoWriter(out_data_path);

% Write data to video file
open(V)
writeVideo(V, D)
close(V)