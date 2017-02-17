#include <iostream>
#include <ctime>
#include "BlockMatching.hpp"

BlockMatching::BlockMatching(std::string path, bool verbose)
{
    this->isVerbose = verbose;
    this->filePath = path;
    this->print("Set filepath to: " + path);
}

BlockMatching::~BlockMatching()
{
    this->print("Object being deleted (DICOM)");
}

void BlockMatching::openCapture()
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

void BlockMatching::captureFrames()
{
    this->openCapture();
    if (this->capture.isOpened())
    {
        cv::Mat frame, grayFrame;

		this->capture.read(frame);

        do
        {
            cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            this->frames.push_back(grayFrame.clone());
			this->capture.read(frame);
        } while (!frame.empty());
    }

    this->capture.release();

    this->print(std::to_string(this->frameCount) + " frames saved from: " + this->filePath);
}

cv::Mat BlockMatching::getFrameFromCaptured(int n)
{
    return this->frames.at(n);
}

void BlockMatching::playFrames(u_int from, u_int to, u_int frameRate)
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

void BlockMatching::print(std::string message)
{
    if (this->isVerbose)
        std::cout << message << std::endl;
}

bool BlockMatching::isRectWithinBounds(cv::Rect b, int width, int height)
{
    return b.x >= 0 && b.y >= 0 && b.x < width && b.y < height;
}

void BlockMatching::sequentialBlockMatch(int macro)
{
    this->openCapture();

    cv::Mat previousFrame, currentFrame, refBlock, nextBlock;

    int macroCenter = macro / 2;    
    double baseSAD = 0, SAD = 0, refSAD = 0;

    time_t tStart = time(0);
	this->capture >> currentFrame;
	cvtColor(currentFrame, currentFrame, cv::COLOR_BGR2GRAY);

    this->print("Starting Sequential Block Search.");

	cv::VideoWriter writer;
	
	int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
	double fps = 25.0;                          // framerate of the created video stream
	std::string filename = "./live.avi";             // name of the output video file
	writer.open(filename, codec, fps, currentFrame.size(), false);

	for (u_int f = 1; f < this->frameCount; ++f) //Offset by 1 because the previous frame is grabbed initially to stop skipping.
    {
        currentFrame.copyTo(previousFrame);
		this->capture >> currentFrame;
		cvtColor(currentFrame, currentFrame, cv::COLOR_BGR2GRAY);
        
        for (u_int x = 0; x < this->width; x += macro)
        {
            for (u_int y = 0; y < this->height; y += macro)
            {
                cv::Rect border(x, y, macro, macro);
                refBlock = previousFrame(border);

				baseSAD = cv::sum(refBlock - cv::abs(currentFrame(border)))[0];
				refSAD = baseSAD;

                cv::Point refCenter(x + macroCenter, y + macroCenter);
                cv::Point endPoint(x + macroCenter, y + macroCenter);

				int searchMacro = macroCenter, x1, y1;
				for (int row = -searchMacro; row <= searchMacro; row++)
				{
					x1 = x + row;
					for (int col = -searchMacro; col <= searchMacro; col++)
					{
						y1 = y + col;

						if (x1 >= 0 && x1 <= this->width - macro && y1 >= 0 && y1 <= this->height - macro) {
							SAD = cv::sum(cv::abs(refBlock - currentFrame(cv::Rect(x1, y1, macro, macro))))[0];

							if (SAD < refSAD && SAD > 10) //MIN THRESH
							{
								refSAD = SAD;
								endPoint.x = x1 + macroCenter;
								endPoint.y = y1 + macroCenter;
							}
						}

					}
				}

				if (baseSAD <= refSAD) {
					endPoint = refCenter;
				}

				//for (int row = 0; row < 200; row++)
				//{
				//	x1 = rowMin + row;
				//	for (int col = 0; col < 200; col++)
				//	{
				//		y1 = colMin + col;

				//		if (x1 < 0 || y1 < 0 || x1 >= (int) this->width - macro || y1 >= (int) this->height - macro)
				//			break;

				//		SAD = cv::sum(cv::abs(refBlock - currentFrame(cv::Rect(x1, y1, macro, macro))))[0];

				//		if (SAD < refSAD && SAD < 1) //MIN THRESH
				//		{
				//			refSAD = SAD;
				//			endPoint.x = x1 + macroCenter;
				//			endPoint.y = y1 + macroCenter;
				//		}
				//	}
				//}

                //cv::rectangle(currentFrame, border, cv::Scalar(20, 20, 20));
                cv::arrowedLine(currentFrame, refCenter, endPoint, cv::Scalar(200, 200, 0));
            }
        }
		//Show where the data was in the last frame
  		cv::imshow(this->filePath, currentFrame);
		cv::waitKey(1);
		std::cout << "FRAME: " << f << "\t\t" << "REMAINING: " << this->frameCount - f << std::endl;
		writer.write(currentFrame);
    }

    std::cout << "FPS (Sequential): " << (double) this->frameCount / difftime(time(0), tStart) << std::endl;
    this->capture.release();
}

void BlockMatching::exhastiveBlockMatch(int macro)
{
	this->openCapture();

	cv::namedWindow(this->filePath, cv::WINDOW_AUTOSIZE);
	cv::Mat referenceFrame, nextFrame, refBlock, nextBlock;

	int macroCenter = macro / 2;
	double baseSAD = 0, SAD = 0, refSAD = 0;

	time_t tStart = time(0);
	this->capture >> nextFrame;

	this->print("Starting Exhastive Block Search.");
	for (u_int f = 0; f < this->frameCount - 1; ++f)
	{
		nextFrame.copyTo(referenceFrame);
		this->capture >> nextFrame;

		for (u_int x = 0; x < this->width; x += macro)
		{
			for (u_int y = 0; y < this->height; y += macro)
			{
				cv::Rect border(x, y, macro, macro);
				refBlock = cv::abs(referenceFrame(border));

				baseSAD = cv::sum(refBlock - cv::abs(nextFrame(border)))[0];
				refSAD = baseSAD;

				cv::Point refCenter(x + macroCenter, y + macroCenter);
				cv::Point endPoint(x + macroCenter, y + macroCenter);

				for (int row = 0, x1 = 0, y1 = 0; row < 3; ++row)
				{
					x1 = (x - macro) + (macro * row);
					for (int col = 0; col < 3; ++col)
					{
						y1 = (y - macro) + (macro * col);

						if (x1 < 0 || y1 < 0 || x1 >= (int) this->width || y1 >= (int) this->height || (row == 1 && col == 1))
							break;

						SAD = cv::sum(nextFrame(cv::Rect(x1, y1, macro, macro)) - refBlock)[0];

						if (SAD < refSAD)
						{
							refSAD = SAD;
							endPoint.x = x1 + macroCenter;
							endPoint.y = y1 + macroCenter;
						}
					}
				}

				cv::rectangle(referenceFrame, border, cv::Scalar(20, 20, 20));
				cv::arrowedLine(referenceFrame, refCenter, baseSAD <= refSAD ? refCenter : endPoint, cv::Scalar(200, 200, 0));
			}
		}

		cv::imshow(this->filePath, referenceFrame);
	}

	std::cout << "FPS (Sequential): " << (double) this->frameCount / difftime(time(0), tStart) << std::endl;
	this->capture.release();
}