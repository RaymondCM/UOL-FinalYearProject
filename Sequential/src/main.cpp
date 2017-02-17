#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "BlockMatching.hpp"

int main(int argc, char **argv)
{
    std::string projectRoot(".");

#ifdef PROJECT_ROOT
    projectRoot = PROJECT_ROOT;
#endif

    std::string dataPath = projectRoot + "/data/input.avi";
    BlockMatching frames(dataPath, true);

    frames.sequentialBlockMatch(50);

    return 0;
}
