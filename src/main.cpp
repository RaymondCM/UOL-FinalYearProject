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

    std::string filePath{srcPath + "/input.avi"};
    DICOM frames(filePath, true);
    //frames.playFrames(0U, 100U);

    //Start Block Matching
    int cols = frames.getCols(), rows = frames.getRows();
    int blockSize = 100, blockCenter = blockSize / 2;
    int frameMax = (int)frames.getFrameCount() - 2U;

    int yM =0 , xM = 0;
    for (int frameMin = 0; frameMin < frameMax; ++frameMin)
    {
        cv::Mat currentFrame = frames.getFrame(frameMin);
        cv::Mat nextFrame = frames.getFrame(frameMin + 1);

        for (int row = 0; row <= (rows - blockSize); row += blockSize)
        {
            cv::Range rowBlock(row, row + blockSize);

            for (int col = 0; col <= (cols - blockSize); col += blockSize)
            {
                cv::Range colBlock(col, col + blockSize);

                double sad0 = cv::sum(currentFrame(rowBlock, colBlock) -
                                      nextFrame(rowBlock, colBlock))[0];

                double bestSad = sad0;

                // if (s < 5000)
                    //currentFrame(rowBlock, rowBlock).setTo(cv::Scalar(50));
                
                cv::Point p1(row+blockCenter, col+blockCenter), p2(row+blockCenter, col+blockCenter);

                if(bestSad >= 0) {
                    cv::circle(currentFrame, p1, blockCenter/2, cv::Scalar(255));
                    cv::arrowedLine(currentFrame, p1,p2, cv::Scalar(255));
                }
                
                // cv::imshow(std::to_string(row)+std::to_string(col), b);
                // cv::waitKey(500);
                // std::cout << "Row:" << rowBlock.start << "-" << rowBlock.end
                //           << "\tCol:" << rowBlock.start << "-" << rowBlock.end
                //           << "\tSAD:" << s << std::endl;
            }
        }
    }

    frames.playFrames(0U, frameMax);

    return 0;
}
