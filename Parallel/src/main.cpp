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

#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include "CLContext.hpp"
#include "Timer.hpp"

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
    std::string dataPath = projectRoot + "/data/input.avi";
	
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
	cv::VideoCapture VC(dataPath);

	//Allocate Mat for previous and current frame
	cv::Mat prev, curr, prevGray, currGray;
	VC >> curr;

	//Define BM parameters
	int width = VC.get(cv::CAP_PROP_FRAME_WIDTH), height = VC.get(cv::CAP_PROP_FRAME_HEIGHT);
	unsigned int blockSize = 24, blockStep = 2, wB = width / blockSize, hB = height / blockSize;
	int bCount = wB * hB;

	//Tell OpenCV to use OpenCL
	cv::ocl::setUseOpenCL(true);

	//Create output Window and use dataPath as unique winname
	cv::namedWindow(dataPath, cv::WINDOW_FULLSCREEN);

	//Should the image file loop?
	bool loop = true;

	//Create Timer to time each frame loop and variables for framerate
	Timer t;
	long long int totalNS = 0;
	unsigned int fCount = 0, updateFreq = 30, currentFrame = 1;

	//Timeout to wait for key press (< 1 Waits indef)
	int cvWaitTime = 1;
	char key;

	try {
		do {
			//Start timer
			t.start();

			curr.copyTo(prev); //TODO: Test skipping frames
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
			cl::Kernel kernel(program, "motion_estimation");
			kernel.setArg(0, prevImage);
			kernel.setArg(1, currImage);
			kernel.setArg(2, blockSize);
			kernel.setArg(3, motionVectors);

			//Queue kernel with global range spanning all blocks
			cl::NDRange global((size_t)wB, (size_t)hB, 1);
			queue.enqueueNDRangeKernel(kernel, 0, global, cl::NullRange);

			//Read motion vector buffer from device
			cl_int2 * mVecBuffer = new cl_int2[bCount];
			queue.enqueueReadBuffer(motionVectors, 0, 0, sizeof(cl_int2) * bCount, mVecBuffer);

			//Draw Motion Vectors from mVecBuffer
			for (size_t i = 0; i < wB; i++)
			{
				for (size_t j = 0; j < hB; j++)
				{
					//Calculate repective position of motion vector
					int id = i + j * wB;

					//Offset drawn point to represent middle rather than top left of block
					cv::Point offset(blockSize / 2, blockSize / 2);
					cv::Point pos(i * blockSize, j * blockSize);
					cv::Point mVec(mVecBuffer[id].x, mVecBuffer[id].y);

					/*cv::rectangle(prev, pos, pos + (offset * 2), cv::Scalar(255, 0, 0, 50));
					cv::arrowedLine(prev, pos + offset, mVec + offset, cv::Scalar(255));*/
					if (pos != mVec) {
						cv::rectangle(prev, pos, pos + cv::Point(blockSize, blockSize), cv::Scalar(255));
						cv::arrowedLine(prev, pos + offset, mVec + offset, cv::Scalar(0, 255, 255));
					}
				}
			}

			//Display current frame on visualisation
			cv::putText(prev, "Frame " + std::to_string(currentFrame) + ", Block Size: " + std::to_string(blockSize), 
				cv::Point(1, height - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255));

			//Display visualisation of motion vectors
			cv::imshow(dataPath, prev);

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

			//Free pointer block
			free(mVecBuffer);
			//*prevBuffer = 0;
			//*currBuffer = 0;

			//End timer
			totalNS += t.end();

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
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << clUtil.GetErrorString(err.err()) << std::endl;
		throw err;
	}

	cv::destroyAllWindows();
	return 0;
}
