% Get File Path
[file_name, root] = uigetfile('*.dcm');
in_data_path = strcat(root, file_name);

% Read in DICOM file
D = dicomread(in_data_path);

% Create VideoWriter Object
[file_name, root] = uiputfile('*.mp4');
out_data_path = strcat(root, file_name);
V = VideoWriter(out_data_path);

% Write data to video file
open(V)
writeVideo(V, D)
close(V)