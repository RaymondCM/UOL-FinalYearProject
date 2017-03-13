#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "Capture.hpp"
#include "Timer.hpp"
#include "Utils.hpp"

int main(int argc, char **argv)
{
	std::string project_root(".");

	#ifdef ROOT_DIR
		project_root = ROOT_DIR;
	#endif

	std::string dataPathVideo = project_root + "/data/input.avi";

	//Open Video Capture to File
	Capture Capture(dataPathVideo);

	//Allocate Mat for previous and current frame
	cv::Mat frame, frame_gray;

	int width = Capture.GetWidth(), height = Capture.GetHeight(), frame_count = Capture.GetFrameCount();

	//Create output Window and use Sequential as unique winname
	std::string winname("Sequential Segmentation");
	cv::namedWindow(winname, cv::WINDOW_FREERATIO);
	cv::setMouseCallback(winname, Util::MouseCallback, NULL);

	//Should the image file loop?
	const bool loop = true;

	//Timeout to wait for key press (< 1 Waits indef)
	int cv_wait_time = 0;
	char key;

	do {
		Capture >> frame;

		//Break if invalid frames and no loop
		if (frame.empty()) {
			//Reset pointer to frame if loop
			if (loop) {
				Capture.SetPos(0);
				Capture >> frame;
				continue;
			}

			break;
		}

		//Convert frames to grayscale for faster processing. Keep original data for visualisation
		cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

		cv::imshow(winname, frame);

		key = (char)cv::waitKey(cv_wait_time);

		if (key == 'p')
			cv_wait_time = cv_wait_time == 0 ? 1 : 0;
	} while (key != 27); //Do while !Esc

	cv::destroyAllWindows();
	return 0;
}
