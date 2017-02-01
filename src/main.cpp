#include <iostream>
#include <string>

#include "DICOM.h"

#include "dcmtk/dcmimgle/dcmimage.h"

int main()
{
    std::string pathRoot;

    #ifdef SOURCE_CODE_LOCATION
        pathRoot = SOURCE_CODE_LOCATION;
    #endif

    std::string fullPath {pathRoot + "/IM_0068-Bmode"};
    DicomImage *image = new DicomImage(fullPath.c_str());

    if(image->getStatus() == EIS_Normal) {
        cout << "Normal Image";
    }

    return 0;
}