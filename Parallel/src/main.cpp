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
	int blockSize = width / 16, wB = width / blockSize, hB = height / blockSize;
	const int bCount = wB * hB;

	//Tell OpenCV to use OpenCL
	cv::ocl::setUseOpenCL(true);

	//Create output Window
	//cv::namedWindow(dataPath, cv::WINDOW_AUTOSIZE);

	//Should the image file loop?
	bool loop = true;

	//Create Timer to time each frame loop
	Timer t;

	try {
		do {
			t.start();
			curr.copyTo(prev); //TODO: Test skipping frames
			VC >> curr;

			if (prev.empty() || curr.empty()) {
				//Reset pointer to frame if loop
				if (loop && VC.isOpened()) {
					VC.set(cv::CAP_PROP_POS_AVI_RATIO, 0);
					VC >> prev;
					VC >> curr;
					continue;
				}

				break;
			}

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
			kernel.setArg(2, (unsigned int)blockSize);
			kernel.setArg(3, motionVectors);

			cl::NDRange global((size_t)wB, (size_t)hB, 1);
			queue.enqueueNDRangeKernel(kernel, 0, global);

			//Read motion vector buffer from device
			cl_int2 * mVecBuffer = new cl_int2[bCount];
			queue.enqueueReadBuffer(motionVectors, 0, 0, sizeof(cl_int2) * bCount, mVecBuffer);

			//Draw Motion Vectors from mVecBuffer
			for (size_t i = 0; i < wB; i++)
			{
				for (size_t j = 0; j < hB; j++)
				{
					int id = i + j * wB;

					//Offset drawn point to represent middle rather than top left of block
					cv::Point offset(blockSize / 2, blockSize / 2);
					cv::Point pos(i * blockSize, j * blockSize);
					cv::Point mVec(mVecBuffer[id].x, mVecBuffer[id].y);

					cv::rectangle(curr, pos, pos + (offset * 2), cv::Scalar(255));
					cv::arrowedLine(curr, pos + offset, mVec + offset, cv::Scalar(255));
				}
			}

			t.end();

			cv::imshow(dataPath, curr);
		} while ((char)cv::waitKey(1) != 27); //Do while !Esc
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << clUtil.GetErrorString(err.err()) << std::endl;
		throw err;
	}

	return 0;
}