#include <string>
#include "DICOM.h"

int main()
{
    std::string srcPath;

    #ifdef SOURCE_CODE_LOCATION
        srcPath = SOURCE_CODE_LOCATION;
    #endif

    DICOM frames(srcPath + "/input.avi", true);

    frames.exhastiveBlockMatch(40);
    return 0;
}
