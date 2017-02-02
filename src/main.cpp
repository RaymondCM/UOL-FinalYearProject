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
    //frames.playFrames(0U, 100U);

    //Start Block Matching
    int cols = frames.getCols(), rows = frames.getRows();
    int bSize = 40, bCenter = bSize / 2;
    int frameMax = (int)frames.getFrameCount() - 2U;

    for (int frameMin = 0; frameMin < frameMax; ++frameMin)
    {
        cv::Mat currentFrame = frames.getFrame(frameMin);
        cv::Mat nextFrame = frames.getFrame(frameMin + 1);

        for (int x = 0; x < cols; x += bSize)
        {
            for (int y = 0; y < rows; y += bSize)
            {
                cv::Rect block(cv::Point(x, y), cv::Size(bSize, bSize));
                cv::Point center(x + bCenter, y + bCenter);

                double baseSAD = cv::sum(nextFrame(block) - currentFrame(block))[0];

                //Calculate SAD of block in first frame to determine baseSAD
                double refSAD = baseSAD;
                cv::Point endPoint = center;

                //Calculate blocks for neighbouring pixels
                //1 2 3
                //4 X 6
                //7 8 9
                int up = y - bSize, down = y + bSize, left = x - bSize, right = x + bSize;

                cv::Rect block1(cv::Point(left, up), cv::Size(bSize, bSize));
                cv::Rect block2(cv::Point(x, up), cv::Size(bSize, bSize));
                cv::Rect block3(cv::Point(right, up), cv::Size(bSize, bSize));

                cv::Rect block4(cv::Point(left, y), cv::Size(bSize, bSize));
                cv::Rect block6(cv::Point(right, y), cv::Size(bSize, bSize));

                cv::Rect block7(cv::Point(left, down), cv::Size(bSize, bSize));
                cv::Rect block8(cv::Point(x, down), cv::Size(bSize, bSize));
                cv::Rect block9(cv::Point(right, down), cv::Size(bSize, bSize));

                double SAD = 0;
                if (validBlock(block1))
                {
                    SAD = cv::sum(nextFrame(block1) - currentFrame(block1))[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block1.x + bSize;
                        endPoint.y = block1.y + bSize;
                    }
                }

                if (validBlock(block2))
                {
                    SAD = cv::sum(nextFrame(block2) - currentFrame(block2))[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block2.x + bCenter;
                        endPoint.y = block2.y + bSize;
                    }
                }

                if (validBlock(block3))
                {
                    SAD = cv::sum(nextFrame(block3) - currentFrame(block3))[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block3.x;
                        endPoint.y = block3.y + bSize;
                    }
                }

                if (validBlock(block4))
                {
                    SAD = cv::sum(nextFrame(block4) - currentFrame(block4))[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block4.x + bSize;
                        endPoint.y = block4.y + bCenter;
                    }
                }

                if (validBlock(block6))
                {
                    SAD = cv::sum(nextFrame(block6) - currentFrame(block6))[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block6.x;
                        endPoint.y = block6.y + bCenter;
                    }
                }

                if (validBlock(block7))
                {
                    SAD = cv::sum(nextFrame(block7) - currentFrame(block7))[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block7.x + bSize;
                        endPoint.y = block7.y;
                    }
                }

                if (validBlock(block8))
                {
                    SAD = cv::sum(nextFrame(block8) - currentFrame(block8))[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block8.x + bCenter;
                        endPoint.y = block8.y;
                    }
                }

                if (validBlock(block9))
                {
                    SAD = cv::sum(nextFrame(block9) - currentFrame(block9))[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block9.x;
                        endPoint.y = block9.y;
                    }
                }

                if (refSAD > -1)
                {
                    cv::circle(nextFrame, center, bCenter, cv::Scalar(32, 12, 67));
                    cv::arrowedLine(nextFrame, center, endPoint, cv::Scalar(255));
                }
            }
        }
    }

    frames.playFrames(0U, frameMax);

    return 0;
}
