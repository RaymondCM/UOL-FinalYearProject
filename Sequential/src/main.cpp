#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "Timer.hpp"

inline float square(float x) {
	return x * x;
}

inline float euclideanDistance(int x2, int x1, int y2, int y1) {
	return sqrt((float)(square(x2 - x1) + square(y2 - y1)));
}

int main(int argc, char **argv)
{
	std::string projectRoot(".");

#ifdef PROJECT_ROOT
	projectRoot = PROJECT_ROOT;
#endif

	std::string dataPath = projectRoot + "/data/input.avi";

	//Open Video Capture to File
	cv::VideoCapture VC(dataPath);

	//Allocate Mat for previous and current frame
	cv::Mat prev, curr, prevGray, currGray;
	VC >> curr;

	//Define BM parameters
	int width = VC.get(cv::CAP_PROP_FRAME_WIDTH), height = VC.get(cv::CAP_PROP_FRAME_HEIGHT);
	unsigned int blockSize = 20, blockStep = 2, wB = width / blockSize, hB = height / blockSize;
	int bCount = wB * hB;

	//Create output Window and use dataPath as unique winname
	cv::namedWindow(dataPath, cv::WINDOW_FREERATIO);

	//Should the image file loop?
	bool loop = true;

	//Create Timer to time each frame loop and variables for framerate
	Timer t;
	long long int totalNS = 0;
	unsigned int fCount = 0, updateFreq = 30, currentFrame = 1;

	//Timeout to wait for key press (< 1 Waits indef)
	int cvWaitTime = 0;
	char key;

	do {
		//Start timer
		t.start();

		prev = curr.clone(); //TODO: Test skipping frames
		VC >> curr;

		//Break if invalid frames and no loop
		if (prev.empty() || curr.empty()) {
			//Reset pointer to frame if loop
			if (loop && VC.isOpened()) {
				VC.set(cv::CAP_PROP_POS_AVI_RATIO, 0);
				VC >> curr;
				currentFrame = 1;
				continue;
			}

			break;
		}

		//Convert frames to grayscale for faster processing. Keep original data for visualisation
		cv::cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(curr, currGray, cv::COLOR_BGR2GRAY);

		//Create point array to store 
		cv::Point * motionVectors = new cv::Point[bCount];

		//Loop over all possible blocks in frame
		for (int x = 0; x < wB; x++) {
			for (int y = 0; y < hB; y++) {
				//Reference point on current frame that will be searched for in the previous frame
				const cv::Point currPoint(x * blockSize, y * blockSize);
				int idx = x + y * wB;

				cv::Rect middle(currPoint.x, currPoint.y, blockSize, blockSize);

				const int sWindow = blockSize;
				float distanceToBlock = FLT_MAX;
				float bestErr = FLT_MAX, err;

				//Loop over all possible blocks within each macroblock
				for (int row = -sWindow; row < sWindow; row++) {
					for (int col = -sWindow; col < sWindow; col++) {
						//Refererence a block to search on the previous frame
						cv::Point refPoint(currPoint.x + row, currPoint.y + col);

						//Check if it lays within the bounds of the capture
						if (refPoint.y >= 0 && refPoint.y < height - blockSize && refPoint.x >= 0 && refPoint.x < width - blockSize) {
							//Calculate SSD (Sum of square differences)
							cv::Mat diff = cv::abs(currGray(middle) - prevGray(cv::Rect(refPoint.x, refPoint.y, blockSize, blockSize)));
							err = cv::sum(diff.mul(diff))[0];

							//Take the lowest error, closeness is preffered.
							float newDistance = euclideanDistance(refPoint.x, currPoint.x, refPoint.y, currPoint.y);

							//Write buffer with the lowest error
							if (err < bestErr) {
								bestErr = err;
								distanceToBlock = newDistance;
								motionVectors[idx] = refPoint;
							}
							else if (err == bestErr && newDistance <= distanceToBlock) {
								distanceToBlock = newDistance;
								motionVectors[idx] = refPoint;
							}
						}
					}
				}
			}
		}

		//End timer so FPS isn't inclusive of drawing onto the screen
		totalNS += t.end();

		cv::Mat display = curr.clone();

		//Draw Motion Vectors
		for (size_t i = 0; i < wB; i++)
		{
			for (size_t j = 0; j < hB; j++)
			{
				//Calculate repective position of motion vector
				int idx = i + j * wB;

				//Offset drawn point to represent middle rather than top left of block
				cv::Point offset(blockSize / 2, blockSize / 2);
				cv::Point pos(i * blockSize, j * blockSize);
				cv::Point mVec(motionVectors[idx].x, motionVectors[idx].y);

				cv::rectangle(display, pos, pos + cv::Point(blockSize, blockSize), cv::Scalar(255));
				cv::arrowedLine(display, pos + offset, mVec + offset, cv::Scalar(0, 255, 255));
			}
		}

		//Free pointer block
		free(motionVectors);

		//Display current frame on visualisation
		cv::putText(display, "Frame " + std::to_string(currentFrame) + ", Block Size: " + std::to_string(blockSize),
			cv::Point(1, height - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255));

		//Display visualisation of motion vectors
		cv::imshow(dataPath, display);

		//Display framerate every updateFreq frames
		if (fCount >= updateFreq) {
			cv::setWindowTitle(dataPath, "Framerate: " + std::to_string(fCount / (totalNS / 1000000000.0)) +
				", Average time between frames [ns]: " + std::to_string(totalNS / fCount));
			totalNS = 0;
			fCount = 0;
		}

		//Increment frame counters
		fCount++;
		currentFrame++;

		key = (char)cv::waitKey(cvWaitTime);

		switch (key) {
		case 'p':
			cvWaitTime = cvWaitTime == 0 ? 1 : 0;
			break;
		case '+':
			blockSize += blockSize < width / 2 ? blockStep : 0;
			wB = width / blockSize;
			hB = height / blockSize;
			bCount = wB * hB;
			break;
		case '-':
			blockSize -= blockSize > blockStep ? blockStep : 0;
			wB = width / blockSize;
			hB = height / blockSize;
			bCount = wB * hB;
			break;
		default:
			break;
		}
	} while (key != 27); //Do while !Esc

	cv::destroyAllWindows();
	return 0;
}
