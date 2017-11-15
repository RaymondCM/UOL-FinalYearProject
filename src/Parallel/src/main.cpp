#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <string>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

//Commented out temporarily to fix VS2017 IDE intelisense error
//TODO: Uncomment this line
//#include "Dicom.hpp"

#include "CLContext.hpp"
#include "Drawing.hpp"
#include "Capture.hpp"
#include "Timer.hpp"
#include "Utils.hpp"
#include "SimpleGraph.hpp"
#include "IO.hpp"

int main(int argc, char **argv)
{

	std::string root_directory(".");
	std::string project_directory(".");

#ifdef ROOT_DIR
	root_directory = ROOT_DIR;
#endif

#ifdef PROJECT_ROOT
	project_directory = PROJECT_ROOT;
#endif

	std::string kernelFile = project_directory + "/opencl/kernels.cl";
	std::string dataPath = root_directory + "/data/IM_0068-Bmode.dcm";
	std::string dataPathVideo = root_directory + "/data/input.avi";
	std::string results_path = root_directory + "/results/raw/parallel/" + std::to_string(std::time(nullptr)) + ".txt";

	//Get Context
	CLContext clUtil(argc, argv);
	cl::Context context = clUtil.GetContext();

	//Load and build kernel file (device code)
	cl::Program::Sources sources;
	clUtil.AddSources(sources, kernelFile);

	//Create program with sources for device
	cl::Program program(context, sources);

	//Attempt to build program
	try {
		program.build();
	}
	catch (cl::Error err) {
		std::cerr << err.what() << std::endl;
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		throw err;
	}

	//Create command queue for context (outside of the loop?)
	cl::CommandQueue queue(context);

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
	int  bID = bSizes.size() >= 2 ? 1 : 0, blockSize = bSizes.at(bID);
	int stepSize = Util::getStepSize(blockSize);

	//Minus one because last block along x * stepSize + y * stepSize * wB will always be out of bounds
	int wB = (width / blockSize * blockSize / stepSize) - 1, hB = (height / blockSize * blockSize / stepSize) - 1;

	int bCount = wB * hB;

	//Tell OpenCV to use OpenCL
	//cv::ocl::setUseOpenCL(true);

	//Create output Window and use Parallel as unique winname
	std::string winname("Parallel");
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
	char key = ' ';
	int method = 0;
	bool draw_motion_vectors = false, draw_hsv = false;

	try {
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
					output_data.NewFile(root_directory + "/results/raw/parallel/" + std::to_string(std::time(nullptr)) + ".txt");
					Capture.SetPos(0);
					Capture >> curr;
					motion_graph.Reset();
					continue;
				}

				break;
			}

			if (set_roi)
				curr = curr(roi);

			//Convert frames to grayscale for faster processing. Keep original data for visualisation
			cv::cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(curr, currGray, cv::COLOR_BGR2GRAY);

			//Convert data (int) to char array
			char *prevBuffer = reinterpret_cast<char *>(prevGray.data);
			char *currBuffer = reinterpret_cast<char *>(currGray.data);

			//Create image2d_t from buffer (CL_INTENSITY = uint4(I,I,I,I))
			//CL_UNSIGNED_INT8 for write_imageui
			cl::ImageFormat fmt(CL_INTENSITY, CL_UNSIGNED_INT8);
			cl::Image2D prevImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fmt, width, height, 0, prevBuffer);
			cl::Image2D currImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fmt, width, height, 0, currBuffer);

			//Create buffer to store motion vectors for blocks of wB * hB (bCount)
			cl::Buffer motionVectors(context, CL_MEM_WRITE_ONLY, sizeof(cl_int2) * bCount);
			cl::Buffer motionDetails(context, CL_MEM_WRITE_ONLY, sizeof(cl_float2) * bCount);

			//Create motion_estimation kernel and set arguments 
			cl::Kernel kernel(program, method == 0 ? "full_exhastive_SAD" : "full_exhastive_ADS");
			kernel.setArg(0, prevImage);
			kernel.setArg(1, currImage);
			kernel.setArg(2, stepSize);
			kernel.setArg(3, blockSize);
			kernel.setArg(4, width);
			kernel.setArg(5, height);
			kernel.setArg(6, motionVectors);
			kernel.setArg(7, motionDetails);

			//Queue kernel with global range spanning all blocks
			cl::NDRange global((size_t)wB, (size_t)hB, 1);
			queue.enqueueNDRangeKernel(kernel, 0, global, cl::NullRange);

			//queue.finish();

			//TODO: declare outside loop
			//Read motion vector buffer from device
			cl_int2 * mVecBuffer = new cl_int2[bCount];
			cl_float2 * mDetailsBuffer = new cl_float2[bCount];

			queue.enqueueReadBuffer(motionVectors, 0, 0, sizeof(cl_int2) * bCount, mVecBuffer);
			queue.enqueueReadBuffer(motionDetails, 0, 0, sizeof(cl_float2) * bCount, mDetailsBuffer);

			//Clock timer so FPS isn't inclusive of drawing onto the screen
			pT.toc();

			//Create seperate file for drawing to the screen
			cv::Mat display = curr.clone();

			//Draw Motion Vectors from mVecBuffer
			cv::Vec4f averages = Util::analyseData(mVecBuffer, mDetailsBuffer, wB * hB);
			motion_graph.AddData(averages[3]);

			//Draw::Arrow(display, cv::Point(averages[0], averages[1]));

			if(draw_motion_vectors)
				Draw::MotionVectors(display, mVecBuffer, wB, hB, blockSize, stepSize);

			if(draw_hsv)
				Draw::MotionVectorHSVAngles(display, mVecBuffer, mDetailsBuffer, wB, hB, blockSize, stepSize, 127, 0.2);

			output_data.AddLine(std::to_string(averages[3]), std::to_string(averages[2]));

			//Free pointer block
			free(mVecBuffer);
			free(mDetailsBuffer);

			//Finish render timer
			rT.toc();

			//Display program information on frame
			//Draw::Text(display, std::to_string(Capture.GetPos()), std::to_string(blockSize), std::to_string(stepSize), std::to_string(pT.getFPSFromElapsed()), std::to_string(rT.getFPSFromElapsed()));
			motion_graph.DrawInfoText(std::to_string(Capture.GetPos()), std::to_string(blockSize), std::to_string(stepSize), std::to_string(pT.getFPSFromElapsed()), std::to_string(rT.getFPSFromElapsed()));
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
			case 'm':
				method = method == 0 ? 1 : 0;
				motion_graph.Reset();
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
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << clUtil.GetErrorString(err.err()) << std::endl;
		throw err;
	}

	cv::destroyAllWindows();
	return 0;
}
