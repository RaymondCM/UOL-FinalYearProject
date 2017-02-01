#include <iostream>
#include <string>
#include <vector>

#include "DICOM.h"

int main()
{
    std::string srcPath;

#ifdef SOURCE_CODE_LOCATION
    srcPath = SOURCE_CODE_LOCATION;
#endif

    std::string filePath {srcPath + "/input.avi"};
    DICOM images(filePath, true);
    
    return 0;
}
