#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "BlockMatching.hpp"
#include "Dicom.hpp"
#include "Capture.hpp"
#include "Timer.hpp"
#include "Utils.hpp"

int main(int argc, char **argv)
{
	std::string projectRoot(".");

#ifdef PROJECT_ROOT
	projectRoot = PROJECT_ROOT;
#endif

	std::string dataPath = projectRoot + "/data/IM_0068-Bmode.dcm";
	std::string dataPathVideo = projectRoot + "/data/input.avi";

	//Open Video Capture to File
	Dicom Capture(dataPath, true);
	//Capture Capture(dataPathVideo);

	//Allocate Mat for previous and current frame
	cv::Mat prev, curr, prevGray, currGray;
	Capture >> curr;

	//Define BM parameters
	int width = Capture.GetWidth(), height = Capture.GetHeight(), frame_count = Capture.GetFrameCount();

	//Get all possible block sizes
	std::vector<int> bSizes = Util::getBlockSizes(width, height);
	int blockSize = bSizes.at(6), bID = 6, wB = width / blockSize, hB = height / blockSize;
	int bCount = wB * hB;

	//Create output Window and use Sequential as unique winname
	std::string winname("Sequential");
	cv::namedWindow(winname, cv::WINDOW_FREERATIO);
	cv::setMouseCallback(winname, Util::MouseCallback, NULL);

	//Should the image file loop?
	bool loop = true;

	//Create Timer to time each frame loop and variables for framerate
	//Log Processed Frames per second and rendered
	Timer pT(30), rT(30);

	//Timeout to wait for key press (< 1 Waits indef)
	int cvWaitTime = 0;
	char key;

	do {
		//Start timer
		pT.tic();
		rT.tic();

		prev = curr.clone(); //TODO: Test skipping frames
		Capture >> curr;

		//Break if invalid frames and no loop
		if (prev.empty() || curr.empty()) {
			//Reset pointer to frame if loop
			if (loop) {
				Capture.SetPos(0);
				Capture >> curr;
				continue;
			}

			break;
		}

		//Convert frames to grayscale for faster processing. Keep original data for visualisation
		cv::cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(curr, currGray, cv::COLOR_BGR2GRAY);

		//Create point array to store 
		cv::Point * motionVectors = new cv::Point[bCount];
		cv::Point2f * motionDetails = new cv::Point2f[bCount];

		//Perform Block Matching
		BlockMatching::FullExhastive(currGray, prevGray, motionVectors, motionDetails, blockSize, width, height, wB, hB);

		//Clock timer so FPS isn't inclusive of drawing onto the screen
		pT.toc();

		cv::Mat display = curr.clone();

		//Draw Motion Vectors from mVecBuffer
		Util::drawMotionVectors(display, motionVectors, wB, hB, blockSize);
		Util::visualiseMotionVectors(display, motionVectors, motionDetails, wB, hB, blockSize, 20, 0.2);

		//Free pointer block
		free(motionVectors);
		free(motionDetails);

		//Finish render timer
		rT.toc();
		

		//Display program information on frame
		Util::drawText(display, std::to_string(Capture.GetPos()), std::to_string(blockSize), std::to_string(pT.getFPSFromElapsed()), std::to_string(rT.getFPSFromElapsed()));

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
