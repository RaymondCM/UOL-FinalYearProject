% Select DICOM file
[in_file_name,in_file_root] = uigetfile('*.dcm','Select the DICOM image file');
in_data_path = strcat(in_file_root, in_file_name);

% Read in DICOM file information
D = dicominfo(in_data_path);

%% Print Information
D