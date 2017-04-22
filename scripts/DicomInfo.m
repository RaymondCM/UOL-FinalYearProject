% Select DICOM file
[file_name, root] = uigetfile('*.dcm');
file_path = strcat(root, file_name);

% Read in DICOM file information and print
D = dicominfo(file_path)