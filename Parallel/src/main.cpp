#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <string>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

//#include "BlockMatching.hpp"
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

    CLContext clUtil(argc, argv);
    clUtil.ListPlatforms();

    try
    {
	cl::Context context = clUtil.GetContext();

	//Create a queue to which we will push commands for the device
	cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

	//Load & build the device code
	cl::Program::Sources sources;
	clUtil.AddSources(sources, kernelFile);

	cl::Program program(context, sources);

	//Build and debug the kernel code
	try
	{
	    program.build();
	}
	catch (const cl::Error &err)
	{
	    std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
	    std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
	    std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
	    throw err;
	}

	//Memory allocation
	//host - input
	std::vector<int> A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; //C++11 allows this type of initialisation
	std::vector<int> B = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

	size_t vector_elements = A.size();	   //number of elements
	size_t vector_size = A.size() * sizeof(int); //size in bytes

	//host - output
	std::vector<int> C(vector_elements);

	//device - buffers
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

	//Part 5 - device operations

	//5.1 Copy arrays A and B to device memory
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);

	//5.2 Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_add = cl::Kernel(program, "add");
	kernel_add.setArg(0, buffer_A);
	kernel_add.setArg(1, buffer_B);
	kernel_add.setArg(2, buffer_C);

	cl::Event prof_event;
	queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

	//5.3 Copy the result from device to host
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);

	std::cout << "Kernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
    }
    catch (cl::Error err)
    {
	std::cerr << "ERROR: " << err.what() << ", " << clUtil.GetErrorString(err.err()) << std::endl;
    }

    return 0;
}
