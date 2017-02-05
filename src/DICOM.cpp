#include <iostream>
#include <ctime>
#include "DICOM.h"

DICOM::DICOM(std::string path, bool verbose)
{
    this->isVerbose = verbose;
    this->filePath = path;
    this->print("Set filepath to: " + path);
}

DICOM::~DICOM()
{
    this->print("Object being deleted (DICOM)");
}

void DICOM::openCapture()
{
    this->print("Attepmting to open capture device.");
    if (this->capture.isOpened())
    {
        this->print("Already open. Setting capture position to 0.");        
        this->capture.set(cv::CAP_PROP_POS_AVI_RATIO, 0);
    }
    else
    {
        this->print("Opening. Initilising frameCount, width and height.");   
        this->capture = cv::VideoCapture(this->filePath);
        this->frameCount = (int) std::round(this->capture.get(cv::CAP_PROP_FRAME_COUNT));
        this->width = (int) std::round(this->capture.get(cv::CAP_PROP_FRAME_WIDTH));
        this->height = (int) std::round(this->capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    }
}

void DICOM::captureFrames()
{
    this->openCapture();
    if (this->capture.isOpened())
    {
        cv::Mat frame, grayFrame;

        do
        {
            this->capture.read(frame);
            cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            this->frames.push_back(grayFrame.clone());
        } while (!frame.empty());
    }

    this->capture.release();

    this->print(std::to_string(this->frameCount) + " frames saved from: " + this->filePath);
}

cv::Mat DICOM::getFrameFromCaptured(int n)
{
    return this->frames.at(n);
}

void DICOM::playFrames(u_int from, u_int to, u_int frameRate)
{
    if (!(from < to && (to < this->frameCount && to > from)))
    {
        this->print("Invalid frame ranges" + from + to);
        return;
    }
    else if (this->frames.empty())
    {
        this->print("Invalid Frame Object");
        return;
    }

    u_int currentFrame = from;
    u_int fpsTimeout = static_cast<u_int>((1000 / frameRate) + 0.5);

    this->print("Playing frames: " + std::to_string(from) + "->" + std::to_string(to));

    std::string videoWindow = this->filePath;
    cv::namedWindow(videoWindow, cv::WINDOW_AUTOSIZE);

    while (currentFrame < to)
    {
        cv::imshow(videoWindow, this->getFrameFromCaptured(currentFrame++));
        cv::setWindowTitle(videoWindow, "Estimated time to completion: " + std::to_string(((to - currentFrame) * fpsTimeout) / 1000) + "s");
        cv::waitKey(fpsTimeout);
    }

    cv::destroyWindow(videoWindow);
}

void DICOM::print(std::string message)
{
    if (this->isVerbose)
        std::cout << message << std::endl;
}

bool DICOM::isRectWithinBounds(cv::Rect b, int width, int height)
{
    return b.x >= 0 && b.y >= 0 && b.x < width && b.y < height;
}

void DICOM::exhastiveBlockMatch(int macro)
{
    this->openCapture();

    cv::namedWindow(this->filePath, cv::WINDOW_AUTOSIZE);
    cv::Mat referenceFrame, nextFrame, refBlock, nextBlock;

    int macroCenter = macro / 2;    
    double baseSAD = 0, SAD = 999999, refSAD = 0;

    time_t tStart = time(0);
    this->capture >> nextFrame;

    this->print("Starting Exhastive Block Search.");
    for (u_int f = 0; f < this->frameCount; ++f)
    {
        referenceFrame = nextFrame.clone();
        this->capture >> nextFrame;

        if(nextFrame.empty())
            break;
        
        for (u_int x = 0; x < this->width; x += macro)
        {
            for (u_int y = 0; y < this->height; y += macro)
            {
                cv::Rect border(x, y, macro, macro);
                refBlock = cv::abs(referenceFrame(border));
                nextBlock = cv::abs(nextFrame(border));

                baseSAD = cv::sum(refBlock - nextBlock)[0];
                refSAD = baseSAD;

                int up = y - macro, down = y + macro, left = x - macro, right = x + macro;

                cv::Point refCenter(x + macroCenter, y + macroCenter);
                cv::Point endPoint(x + macroCenter, y + macroCenter);

                ///TODO: Major Refactor for efficiancy
                cv::Rect block1(cv::Point(left, up), cv::Size(macro, macro));
                cv::Rect block2(cv::Point(x, up), cv::Size(macro, macro));
                cv::Rect block3(cv::Point(right, up), cv::Size(macro, macro));

                cv::Rect block4(cv::Point(left, y), cv::Size(macro, macro));
                cv::Rect block6(cv::Point(right, y), cv::Size(macro, macro));

                cv::Rect block7(cv::Point(left, down), cv::Size(macro, macro));
                cv::Rect block8(cv::Point(x, down), cv::Size(macro, macro));
                cv::Rect block9(cv::Point(right, down), cv::Size(macro, macro));

                if (this->isRectWithinBounds(block1, this->width, this->height))
                {
                    SAD = cv::sum(nextFrame(block1) - refBlock)[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block1.x + macro;
                        endPoint.y = block1.y + macro;
                    }
                }

                if (this->isRectWithinBounds(block2, this->width, this->height))
                {
                    SAD = cv::sum(nextFrame(block2) - refBlock)[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block2.x + macroCenter;
                        endPoint.y = block2.y + macro;
                    }
                }

                if (this->isRectWithinBounds(block3, this->width, this->height))
                {
                    SAD = cv::sum(nextFrame(block3) - refBlock)[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block3.x;
                        endPoint.y = block3.y + macro;
                    }
                }

                if (this->isRectWithinBounds(block4, this->width, this->height))
                {
                    SAD = cv::sum(nextFrame(block4) - refBlock)[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block4.x + macro;
                        endPoint.y = block4.y + macroCenter;
                    }
                }

                if (this->isRectWithinBounds(block6, this->width, this->height))
                {
                    SAD = cv::sum(nextFrame(block6) - refBlock)[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block6.x;
                        endPoint.y = block6.y + macroCenter;
                    }
                }

                if (this->isRectWithinBounds(block7, this->width, this->height))
                {
                    SAD = cv::sum(nextFrame(block7) - refBlock)[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block7.x + macro;
                        endPoint.y = block7.y;
                    }
                }

                if (this->isRectWithinBounds(block8, this->width, this->height))
                {
                    SAD = cv::sum(nextFrame(block8) - refBlock)[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block8.x + macroCenter;
                        endPoint.y = block8.y;
                    }
                }

                if (this->isRectWithinBounds(block9, this->width, this->height))
                {
                    SAD = cv::sum(nextFrame(block9) - refBlock)[0];
                    if (SAD < refSAD)
                    {
                        refSAD = SAD;
                        endPoint.x = block9.x;
                        endPoint.y = block9.y;
                    }
                }

                if (baseSAD <= refSAD)
                {
                    endPoint = refCenter;
                }

                cv::rectangle(referenceFrame, border, cv::Scalar(55,55,55, 0));
                cv::arrowedLine(referenceFrame, refCenter, endPoint, cv::Scalar(255,255,255));
            }
        }

        cv::imshow(this->filePath, referenceFrame);
    }

    std::cout << "FPS (Exhastive Block Search): " << (double) this->frameCount / difftime(time(0), tStart) << std::endl;

    this->capture.release();
}