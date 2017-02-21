#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <fstream>
#include <vector>

//Utility Class for OpenCL Platforms and Devices based on Utils.h provided by http://staff.lincoln.ac.uk/gcielniak
//TODO: Change from class to namespace.
class CLContext
{
  public:
    CLContext()
    {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (size_t i = 0; i < platforms.size(); i++)
	{
	    platformDevices.push_back(std::make_pair(platforms[i], std::vector<cl::Device>()));
	    platforms[i].getDevices((cl_device_type)CL_DEVICE_TYPE_ALL, &platformDevices[i].second);
	}
    };

    CLContext(int argc, char **argv) : CLContext()
    {
	InitialiseArguments(argc, argv);
    };

    void ListPlatforms()
    {
	std::cout << platformDevices.size() << " Available Platform(s):" << std::endl;

	for (size_t i = 0; i < platformDevices.size(); i++)
	{
	    cl::Platform &platform = platformDevices[i].first;
	    std::vector<cl::Device> &devices = platformDevices[i].second;

	    std::cout << "\nPlatform " << i << " (" << devices.size() << " device(s)): " << platform.getInfo<CL_PLATFORM_NAME>()
		      << ": " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

	    for (size_t j = 0; j < devices.size(); j++)
	    {
		std::cout << "\n\tDevice " << j << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << ": "
			  << devices[j].getInfo<CL_DEVICE_VERSION>() << std::endl;

		cl_device_type device_type = devices[j].getInfo<CL_DEVICE_TYPE>();
		std::cout << "\t\tType: ";

		//Bitwise AND to determine type (0 for No Match)
		if (device_type & CL_DEVICE_TYPE_DEFAULT)
		    std::cout << "DEFAULT";
		if (device_type & CL_DEVICE_TYPE_CPU)
		    std::cout << "CPU";
		if (device_type & CL_DEVICE_TYPE_GPU)
		    std::cout << "GPU";
		if (device_type & CL_DEVICE_TYPE_ACCELERATOR)
		    std::cout << "ACCELERATOR";

		std::cout << "\n\t\tCompute Units: " << devices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		std::cout << "\n\t\tClock Frequency (MHz): " << devices[j].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
		std::cout << "\n\t\tMax Memory | Max Allocatable (B): " << devices[j].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << " | "
			  << devices[j].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
	    }

	    std::cout << std::endl;
	}
    };

    cl::Context GetContext()
    {
	platformName = platformDevices[platformID].first.getInfo<CL_PLATFORM_NAME>();
	deviceName = platformDevices[platformID].second[deviceID].getInfo<CL_DEVICE_NAME>();
	std::cout << "Context: " << platformName << "( " << deviceName << ")" << std::endl;
	return cl::Context({platformDevices[platformID].second[deviceID]});
    };

    std::string GetPlatformName()
    {
	return platformName;
    };

    std::string GetDeviceName()
    {
	return deviceName;
    };

    void InitialiseArguments(int argc, char **argv)
    {
	for (int i = 1; i < argc; i++)
	{
	    if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1)))
	    {
		platformID = atoi(argv[++i]);
	    }
	    else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1)))
	    {
		deviceID = atoi(argv[++i]);
	    }
	    else if (strcmp(argv[i], "-l") == 0)
	    {
		ListPlatforms();
	    }
	    else if (strcmp(argv[i], "-h") == 0)
	    {
		PrintArgumentsHelp();
	    }
	}
    }

    void PrintArgumentsHelp()
    {
	std::cerr << "USAGE:" << std::endl;
	std::cerr << "\t-p <platform_id> : Select Platform." << std::endl;
	std::cerr << "\t-d <device_id> : Select Device." << std::endl;
	std::cerr << "\t-l : List System Platform and Devices." << std::endl;
	std::cerr << "\t-h : Print Arguments Help." << std::endl;
    }

    void AddSources(cl::Program::Sources &sources, const std::string &filePath)
    {
	std::ifstream kFile(filePath);

	if (!kFile.good())
	{
	    std::runtime_error e(("Kernel File (" + filePath + ") does not exist/you do not have access.").c_str());
	    std::cerr << e.what() << std::endl;
	    throw e;
	}

	std::string *kernel = new std::string(std::istreambuf_iterator<char>(kFile), (std::istreambuf_iterator<char>()));
	sources.push_back(std::make_pair((*kernel).c_str(), kernel->length() + 1));
    }

    const char *GetErrorString(cl_int error)
    {
	switch (error)
	{
	//Runtime and JIT Errors
	case 0:
	    return "CL_SUCCESS";
	case -1:
	    return "CL_DEVICE_NOT_FOUND";
	case -2:
	    return "CL_DEVICE_NOT_AVAILABLE";
	case -3:
	    return "CL_COMPILER_NOT_AVAILABLE";
	case -4:
	    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5:
	    return "CL_OUT_OF_RESOURCES";
	case -6:
	    return "CL_OUT_OF_HOST_MEMORY";
	case -7:
	    return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8:
	    return "CL_MEM_COPY_OVERLAP";
	case -9:
	    return "CL_IMAGE_FORMAT_MISMATCH";
	case -10:
	    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11:
	    return "CL_BUILD_PROGRAM_FAILURE";
	case -12:
	    return "CL_MAP_FAILURE";
	case -13:
	    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14:
	    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15:
	    return "CL_COMPILE_PROGRAM_FAILURE";
	case -16:
	    return "CL_LINKER_NOT_AVAILABLE";
	case -17:
	    return "CL_LINK_PROGRAM_FAILURE";
	case -18:
	    return "CL_DEVICE_PARTITION_FAILED";
	case -19:
	    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	//Compile Errors
	case -30:
	    return "CL_INVALID_VALUE";
	case -31:
	    return "CL_INVALID_DEVICE_TYPE";
	case -32:
	    return "CL_INVALID_PLATFORM";
	case -33:
	    return "CL_INVALID_DEVICE";
	case -34:
	    return "CL_INVALID_CONTEXT";
	case -35:
	    return "CL_INVALID_QUEUE_PROPERTIES";
	case -36:
	    return "CL_INVALID_COMMAND_QUEUE";
	case -37:
	    return "CL_INVALID_HOST_PTR";
	case -38:
	    return "CL_INVALID_MEM_OBJECT";
	case -39:
	    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40:
	    return "CL_INVALID_IMAGE_SIZE";
	case -41:
	    return "CL_INVALID_SAMPLER";
	case -42:
	    return "CL_INVALID_BINARY";
	case -43:
	    return "CL_INVALID_BUILD_OPTIONS";
	case -44:
	    return "CL_INVALID_PROGRAM";
	case -45:
	    return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46:
	    return "CL_INVALID_KERNEL_NAME";
	case -47:
	    return "CL_INVALID_KERNEL_DEFINITION";
	case -48:
	    return "CL_INVALID_KERNEL";
	case -49:
	    return "CL_INVALID_ARG_INDEX";
	case -50:
	    return "CL_INVALID_ARG_VALUE";
	case -51:
	    return "CL_INVALID_ARG_SIZE";
	case -52:
	    return "CL_INVALID_KERNEL_ARGS";
	case -53:
	    return "CL_INVALID_WORK_DIMENSION";
	case -54:
	    return "CL_INVALID_WORK_GROUP_SIZE";
	case -55:
	    return "CL_INVALID_WORK_ITEM_SIZE";
	case -56:
	    return "CL_INVALID_GLOBAL_OFFSET";
	case -57:
	    return "CL_INVALID_EVENT_WAIT_LIST";
	case -58:
	    return "CL_INVALID_EVENT";
	case -59:
	    return "CL_INVALID_OPERATION";
	case -60:
	    return "CL_INVALID_GL_OBJECT";
	case -61:
	    return "CL_INVALID_BUFFER_SIZE";
	case -62:
	    return "CL_INVALID_MIP_LEVEL";
	case -63:
	    return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64:
	    return "CL_INVALID_PROPERTY";
	case -65:
	    return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66:
	    return "CL_INVALID_COMPILER_OPTIONS";
	case -67:
	    return "CL_INVALID_LINKER_OPTIONS";
	case -68:
	    return "CL_INVALID_DEVICE_PARTITION_COUNT";
	//Extension Errors
	case -1000:
	    return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001:
	    return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002:
	    return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003:
	    return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004:
	    return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005:
	    return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default:
	    return "Unknown OpenCL error";
	}
    };

  private:
    std::vector<std::pair<cl::Platform, std::vector<cl::Device>>> platformDevices;
    std::string platformName = "", deviceName = "";
    int platformID = 0, deviceID = 0;
};