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

	//Open Video Capture to File
	cv::VideoCapture VC(dataPath);

	//Allocate Mat for previous and current frame
	cv::Mat prev, curr;
	VC >> prev;
	VC >> curr;

	//Define BM parameters
	int width = VC.get(cv::CAP_PROP_FRAME_WIDTH), height = VC.get(cv::CAP_PROP_FRAME_HEIGHT);
	int blockSize = width / 16, wB = width / blockSize, hB = height / blockSize;

	//Tell OpenCV to use OpenCL
	cv::ocl::setUseOpenCL(true);

	//Create output Window
	//cv::namedWindow(dataPath, cv::WINDOW_AUTOSIZE);

	//Should the image file loop?
	bool loop = true;

	try {

		do {
			VC >> prev; //TODO: DOnt skip frames
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

			cvtColor(prev, prev, cv::COLOR_BGR2GRAY);
			cvtColor(curr, curr, cv::COLOR_BGR2GRAY);

			//Create command queue for context (outside of the loop?)
			cl::CommandQueue queue(context);

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

			char *prevBuffer = reinterpret_cast<char *>(prev.data);
			char *currBuffer = reinterpret_cast<char *>(curr.data);

			cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
			cl::ImageFormat fmt(CL_INTENSITY, CL_UNSIGNED_INT8);

			cl::Image2D prevImage(context, flags, fmt, width, height, 0, prevBuffer);
			cl::Image2D currImage(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, fmt, width, height, 0, currBuffer);


			//Create shift kernel
			cl::Kernel kernel(program, "shift");
			kernel.setArg(0, prevImage);
			kernel.setArg(1, (unsigned int) blockSize);
			kernel.setArg(2, currImage);

			queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange((size_t)wB, (size_t)hB, 1));

			// Download the dst data from the device (?)
			//cv::Mat mat_dst = uOut.getMat(cv::ACCESS_READ);
			cl::size_t<3> origin;
			origin[0] = 0;
			origin[1] = 0;
			origin[2] = 0;
			cl::size_t<3> region;
			region[0] = (size_t)width;
			region[1] = (size_t)height;
			region[2] = 1;

			queue.enqueueReadImage(currImage, 0, origin, region, 0, 0, currBuffer);

			cv::imshow(dataPath, cv::Mat(height, width, CV_8UC1, &currBuffer[0]));
		} while ((char)cv::waitKey(100) != 27); //Do while !Esc
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << clUtil.GetErrorString(err.err()) << std::endl;
		throw err;
	}

	return 0; 
    try
    {
        cv::VideoCapture V(dataPath);
        //cv::Mat prev, curr;
        cv::Mat prev, curr;
        V >> prev;
        V >> curr;

        int width = V.get(cv::CAP_PROP_FRAME_WIDTH), height = V.get(cv::CAP_PROP_FRAME_HEIGHT);
        cvtColor(prev, prev, cv::COLOR_BGR2GRAY);
        cvtColor(curr, curr, cv::COLOR_BGR2GRAY);

        cl_int res = CL_SUCCESS;
        cl_uint num_entries = 0;

        res = clGetPlatformIDs(0, 0, &num_entries);
        if (CL_SUCCESS != res)
            return -1;

        std::cout << "Have OpenCL?: " << cv::ocl::haveOpenCL() << std::endl;

        // cv::imshow("PREV", prev);
        // cv::imshow("CURR", curr);
        // cv::waitKey(50000);

        if (!cv::ocl::haveOpenCL())
        {
            std::cout << "OpenCL is not avaiable..." << std::endl;
            return -1;
        }

        cv::ocl::Context context;
        if (!context.create(cv::ocl::Device::TYPE_GPU))
        {
            std::cout << "Failed creating the context..." << std::endl;
            return -1;
        }

        // In OpenCV 3.0.0 beta, only a single device is detected.
        std::cout << context.ndevices() << " GPU devices are detected." << std::endl;
        for (int i = 0; i < context.ndevices(); i++)
        {
            cv::ocl::Device device = context.device(i);
            std::cout << "name                 : " << device.name() << std::endl;
            std::cout << "available            : " << device.available() << std::endl;
            std::cout << "imageSupport         : " << device.imageSupport() << std::endl;
            std::cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << std::endl;
            std::cout << std::endl;
        }

        // Select the first device
        cv::ocl::Device(context.device(0));

        // Transfer Mat data to the device
        cv::Mat mat_src = prev;
        mat_src.convertTo(mat_src, CV_32F, 1.0 / 255);
        cv::UMat umat_src = mat_src.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat umat_dst(curr.size(), CV_32F, cv::ACCESS_WRITE, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        std::ifstream ifs(kernelFile);
        if (ifs.fail())
            return -1;
        std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        cv::ocl::ProgramSource programSource(kernelSource);

        // Compile the kernel code
        cv::String errmsg;
        cv::String buildopt = cv::format("-D dstT=%s", cv::ocl::typeToStr(umat_dst.depth())); // "-D dstT=float"
        cv::ocl::Program program = context.getProg(programSource, buildopt, errmsg);

        cv::ocl::Image2D image(umat_src);
        float shift_x = -100.5;
        float shift_y = -50.0;
        cv::ocl::Kernel kernel("shift", program);
        kernel.args(image, shift_x, shift_y, cv::ocl::KernelArg::ReadWrite(umat_dst));

        size_t globalThreads[3] = {(size_t)mat_src.cols, (size_t)mat_src.rows, 1};
        //size_t localThreads[3] = { 16, 16, 1 };
        bool success = kernel.run(3, globalThreads, NULL, true);

        if (!success)
        {
            std::cout << "Failed running the kernel..." << std::endl;
            return -1;
        }

        // Download the dst data from the device (?)
        cv::Mat mat_dst = umat_dst.getMat(cv::ACCESS_READ);

        cv::imshow("src", mat_src);
        cv::imshow("dst", mat_dst);
        cv::waitKey();
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", " << clUtil.GetErrorString(err.err()) << std::endl;
    }

    return 0;
}
