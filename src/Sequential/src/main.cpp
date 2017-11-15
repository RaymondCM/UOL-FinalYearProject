#include <iostream>
#include <string>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

//Commented out temporarily to fix VS2017 IDE intelisense error
//TODO: Uncomment this line
//#include "Dicom.hpp"

#include "BlockMatching.hpp"
#include "Drawing.hpp"
#include "Capture.hpp"
#include "Timer.hpp"
#include "Utils.hpp"
#include "SimpleGraph.hpp"
#include "IO.hpp"

int main(int argc, char **argv)
{
	std::string root_directory(".");

	#ifdef ROOT_DIR
		root_directory = ROOT_DIR;
	#endif

	std::string dataPath = root_directory + "/data/IM_0068-Bmode.dcm";
	std::string dataPathVideo = root_directory + "/data/input.avi";
	std::time_t t = std::time(nullptr);
	std::string results_path = root_directory + "/results/raw/sequential/" + std::to_string(std::time(nullptr)) + ".txt";

	//Open Video Capture to File
	//Dicom Capture(dataPath, true);
	Capture Capture(dataPathVideo);

	//Allocate Mat for previous and current frame
	cv::Mat prev, curr, prevGray, currGray;
	Capture >> curr;

	//Select ROI
	cv::Rect roi;
	bool set_roi = true;

	if (set_roi)
	{
		std::string winname("Press' Y' or 'y' when ROI selection has been made");
		cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);

		cv::imshow(winname, curr);
		cv::setMouseCallback(winname, Util::ROIMouseCallback, nullptr);
		roi = Util::WaitForROI(winname, curr);

		cv::destroyWindow(winname);
		curr = curr(roi);
	}

	//Define BM parameters
	int width = curr.size().width, height = curr.size().height, frame_count = Capture.GetFrameCount();

	//Get all possible block sizes
	std::vector<int> bSizes = Util::getBlockSizes(width, height);
	int bID = bSizes.size() >= 2 ? 1 : 0, blockSize = bSizes.at(bID);
	int stepSize = Util::getStepSize(blockSize);

	//Minus one because last block along x * stepSize + y * stepSize * wB will always be out of bounds
	int wB = (width / blockSize * blockSize / stepSize) - 1, hB = (height / blockSize * blockSize / stepSize) - 1;

	int bCount = wB * hB;

	//Create output Window and use Sequential as unique winname
	std::string winname("Sequential");
	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);

	//Should the image file loop?
	bool loop = true;

	//Create Timer to time each frame loop and variables for framerate
	//Log Processed Frames per second and rendered
	Timer pT(50), rT(50);

	//Create Real-Time graph to display average angular motion
	SimpleGraph motion_graph(1024, 512, 128);

	//Create File Writer
	IO::Writer output_data(results_path);
	output_data.AddLine("Angle 0-360", "Magnitude");

	//Timeout to wait for key press (< 1 Waits indef)
	int cvWaitTime = 1;
	char key;

	bool draw_motion_vectors = false, draw_hsv = false;

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
				output_data.Write();
				output_data.NewFile(root_directory + "/results/raw/sequential/" + std::to_string(std::time(nullptr)) + ".txt");
				Capture.SetPos(0);
				Capture >> curr;
				continue;
			}

			break;
		}

		if (set_roi)
			curr = curr(roi);

		//Convert frames to grayscale for faster processing. Keep original data for visualisation
		cv::cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(curr, currGray, cv::COLOR_BGR2GRAY);

		//Create point array to store 
		cv::Point * motionVectors = new cv::Point[bCount];
		cv::Point2f * motionDetails = new cv::Point2f[bCount];

		//Perform Block Matching
		BlockMatching::FullExhastiveSAD(currGray, prevGray, motionVectors, motionDetails, blockSize, stepSize, width, height, wB, hB);

		//Clock timer so FPS isn't inclusive of drawing onto the screen
		pT.toc();

		cv::Mat display = curr.clone();

		//Draw Motion Vectors from mVecBuffer

		cv::Vec4f averages = Util::analyseData(motionVectors, motionDetails, wB * hB);
		motion_graph.AddData(averages[3]);
		//Draw::Arrow(display, cv::Point(averages[0], averages[1]));

		if(draw_motion_vectors)
			Draw::MotionVectors(display, motionVectors, wB, hB, blockSize, stepSize);

		if(draw_hsv)
			Draw::MotionVectorHSVAngles(display, motionVectors, motionDetails, wB, hB, blockSize, stepSize, 127, 0.2);

		output_data.AddLine(std::to_string(averages[3]), std::to_string(averages[2]));

		//Free pointer block
		free(motionVectors);
		free(motionDetails);

		//Finish render timer
		rT.toc();

		//Display program information on frame
		Draw::Text(display, std::to_string(Capture.GetPos()), std::to_string(blockSize),
			std::to_string(stepSize), std::to_string(pT.getFPSFromElapsed()), std::to_string(rT.getFPSFromElapsed()));

		//Display visualisation of motion vectors
		cv::imshow(winname, display);
		motion_graph.Show();

		key = (char)cv::waitKey(cvWaitTime);

		switch (key) {
		case 'p':
			cvWaitTime = cvWaitTime == 0 ? 1 : 0;
			break;
		case '+':
			bID = bID < bSizes.size() - 1 ? bID + 1 : bID;
			blockSize = bSizes.at(bID);
			stepSize = Util::getStepSize(blockSize);
			wB = (width / blockSize * blockSize / stepSize) - 1;
			hB = (height / blockSize * blockSize / stepSize) - 1;
			bCount = wB * hB;
			break;
		case '-':
			bID = bID > 0 ? bID - 1 : 0;
			blockSize = bSizes.at(bID);
			stepSize = Util::getStepSize(blockSize);
			wB = (width / blockSize * blockSize / stepSize) - 1;
			hB = (height / blockSize * blockSize / stepSize) - 1;
			bCount = wB * hB;
			break;
		case 'd':
			draw_motion_vectors = !draw_motion_vectors;
			break;
		case 'h':
			draw_hsv = !draw_hsv;
			break;
		default:
			break;
		}
	} while (key != 27); //Do while !Esc

	cv::destroyAllWindows();
	return 0;
}
