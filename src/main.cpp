#include <iostream>
#include <string>
#include <vector>

#include "DICOM.h"

bool validBlock(cv::Rect b, int cols = 800, int rows = 600)
{
    return b.x >= 0 && b.y >= 0 && b.x < cols && b.y < rows;
}

int main()
{
    std::string srcPath;

#ifdef SOURCE_CODE_LOCATION
    srcPath = SOURCE_CODE_LOCATION;
#endif

    std::string filePath{srcPath + "/input.avi"};
    DICOM frames(filePath, true);

    frames.exhastiveBlockMatch(40);
    return 0;
}
