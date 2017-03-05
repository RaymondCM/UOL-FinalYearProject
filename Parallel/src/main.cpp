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

#include "CLContext.hpp"
#include "Dicom.hpp"
#include "Capture.hpp"
#include "Timer.hpp"
#include "Utils.hpp"

int main(int argc, char **argv)
{
    std::string projectRoot(".");
    std::string targetRoot(".");

	#ifdef PROJECT_ROOT
		projectRoot = PROJECT_ROOT;
	#endif

	#ifdef TARGET_ROOT
		targetRoot = TARGET_ROOT;
	#endif

    std::string kernelFile = targetRoot + "/opencl/kernels.cl";
	std::string dataPath = projectRoot + "/data/IM_0068-Bmode.dcm";
	std::string dataPathVideo = projectRoot + "/data/input.avi";
	
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
	Dicom Capture(dataPath, true);
	//Capture Capture(dataPathVideo);

	//Allocate Mat for previous and current frame
	cv::Mat prev, curr, prevGray, currGray;
	Capture >> curr;

	//Define BM parameters
	int width = Capture.GetWidth(), height = Capture.GetHeight();

	//Get all possible block sizes
	std::vector<int> bSizes = Util::getBlockSizes(width, height);
	int blockSize = bSizes.at(6), bID = 6, wB = width / blockSize, hB = height / blockSize;
	int bCount = wB * hB;

	//Tell OpenCV to use OpenCL
	//cv::ocl::setUseOpenCL(true);

	//Create output Window and use Parallel as unique winname
	std::string winname("Parallel");
	cv::namedWindow(winname, cv::WINDOW_FREERATIO);

	//Should the image file loop?
	bool loop = true;

	//Create Timer to time each frame loop and variables for framerate
	Timer t(30);

	//Timeout to wait for key press (< 1 Waits indef)
	int cvWaitTime = 1;
	char key;

	try {
		do {
			//Start timer
			t.tic();

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

			//Create motion_estimation kernel and set arguments 
			cl::Kernel kernel(program, "full_exhastive");
			kernel.setArg(0, prevImage);
			kernel.setArg(1, currImage);
			kernel.setArg(2, blockSize);
			kernel.setArg(3, width);
			kernel.setArg(4, height);
			kernel.setArg(5, motionVectors);

			//Queue kernel with global range spanning all blocks
			cl::NDRange global((size_t)wB, (size_t)hB, 1);
			queue.enqueueNDRangeKernel(kernel, 0, global, cl::NullRange);
			
			//queue.finish();

			//Read motion vector buffer from device
			cl_int2 * mVecBuffer = new cl_int2[bCount];
			queue.enqueueReadBuffer(motionVectors, 0, 0, sizeof(cl_int2) * bCount, mVecBuffer);

			//Clock timer so FPS isn't inclusive of drawing onto the screen
			t.toc();

			//Create seperate file for drawing to the screen
			cv::Mat display = curr.clone();

			//Draw Motion Vectors from mVecBuffer
			Util::drawMotionVectors(display, mVecBuffer, wB, hB, blockSize);

			//Free pointer block
			free(mVecBuffer);

			//Display program information on frame
			Util::drawText(display, std::to_string(Capture.GetPos()), std::to_string(blockSize), std::to_string(t.getFPSFromElapsed()));

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
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << clUtil.GetErrorString(err.err()) << std::endl;
		throw err;
	}

	cv::destroyAllWindows();
	return 0;
}
