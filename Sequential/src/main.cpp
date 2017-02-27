#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "BlockMatching.hpp"
#include "Capture.hpp"
#include "Timer.hpp"
#include "Utils.hpp"

int main(int argc, char **argv)
{
	std::string projectRoot(".");

#ifdef PROJECT_ROOT
	projectRoot = PROJECT_ROOT;
#endif

	std::string dataPath = projectRoot + "/data/input.avi";

	//Open Video Capture to File
	Capture VC(dataPath);

	//Allocate Mat for previous and current frame
	cv::Mat prev, curr, prevGray, currGray;
	VC >> curr;

	//Define BM parameters
	int width = VC.width(), height = VC.height();

	//Get all possible block sizes
	std::vector<int> bSizes = Util::getBlockSizes(width, height);
	int blockSize = bSizes.at(6), bID = 6, wB = width / blockSize, hB = height / blockSize;
	int bCount = wB * hB;

	//Create output Window and use Sequential as unique winname
	std::string winname("Sequential");
	cv::namedWindow(winname, cv::WINDOW_FREERATIO);

	//Should the image file loop?
	bool loop = true;

	//Create Timer to time each frame loop and variables for framerate
	Timer t(30);

	//Timeout to wait for key press (< 1 Waits indef)
	int cvWaitTime = 0;
	char key;

	do {
		//Start timer
		t.tic();

		prev = curr.clone(); //TODO: Test skipping frames
		VC >> curr;

		//Break if invalid frames and no loop
		if (prev.empty() || curr.empty()) {
			//Reset pointer to frame if loop
			if (loop && VC.isOpened()) {
				VC.reset();
				VC >> curr;
				continue;
			}

			break;
		}

		//Convert frames to grayscale for faster processing. Keep original data for visualisation
		cv::cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(curr, currGray, cv::COLOR_BGR2GRAY);

		//Create point array to store 
		cv::Point * motionVectors = new cv::Point[bCount];

		//Perform Block Matching
		BlockMatching::FullExhastive(currGray, prevGray, motionVectors, blockSize, width, height, wB, hB);

		//Clock timer so FPS isn't inclusive of drawing onto the screen
		t.toc();

		cv::Mat display = curr.clone();

		//Draw Motion Vectors from mVecBuffer
		Util::drawMotionVectors(display, motionVectors, wB, hB, blockSize);

		//Free pointer block
		free(motionVectors);

		//Display program information on frame
		Util::drawText(display, std::to_string(VC.pos()), std::to_string(blockSize), std::to_string(t.getFPSFromElapsed()));

		//Display visualisation of motion vectors
		cv::imshow(winname, display);

		key = (char)cv::waitKey(cvWaitTime);

		switch (key) {
		case 'p':
			cvWaitTime = cvWaitTime == 0 ? 1 : 0;
			break;
		case '+':
			bID = bID < bSizes.size() - 1 ? bID + 1 : bID;
			blockSize = bSizes.at(bID);
			wB = width / blockSize;
			hB = height / blockSize;
			bCount = wB * hB;
			break;
		case '-':
			bID = bID > 0 ? bID - 1 : 0;
			blockSize = bSizes.at(bID);
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
