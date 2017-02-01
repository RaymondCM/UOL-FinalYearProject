#include <iostream>
#include "DICOM.h"

using namespace std;

DICOM::DICOM(string path)
{
    this->setFilePath(path);
}

void DICOM::setFilePath(string path)
{
    this->filePath = path;
    cout << "Set file path to: '" << path << "'" << endl;
}
