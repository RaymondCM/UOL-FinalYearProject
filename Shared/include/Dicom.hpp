#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmimage/diregist.h"
#include "dcmtk/dcmjpeg/djencode.h"
#include "dcmtk/dcmjpeg/djdecode.h" 
#include "dcmtk/dcmdata/dcxfer.h"

class Dicom {
public:
	Dicom(std::string filePath, bool compressed = true) {
		//Register codecs for dcmtk until class is destroyed
		DJDecoderRegistration::registerCodecs();
		this->path = filePath;
		this->compressed = compressed;
		this->Load();
	};

	~Dicom() {
		DJDecoderRegistration::cleanup();
	};

	//Functions that will eventually allow eassier use of this class
	bool Load() {
		if (this->compressed) {
			return LoadCompressedFile();
		}
		else {
			return LoadDecompressedFile();
		}
	};

	//Functions that will eventually allow eassier use of this class
	bool GetFrame(cv::Mat& in, int index) {
		if (this->compressed) {
			return GetCompressedFrame(in, index);
		}
		else {
			return GetDecompressedFrame(in, index);
		}
	};

	cv::Mat& operator>> (cv::Mat& in)
	{
		if (this->frame_index >= this->frame_count)
			this->frame_index = 0;

		this->GetFrame(in, this->frame_index);
		this->frame_index++;
		return in;
	};

	//TODO: Fully Implement Compressed/Decompressed Files for speed (3x faster)
	bool LoadDecompressedFile() {
		this->frames = new DicomImage(this->path.c_str(), EXS_LittleEndianExplicit);
		this->width = (long int)this->frames->getWidth();
		this->height = (long int)this->frames->getHeight();
		this->bits_allocated = (long int)this->frames->getDepth();
		this->frame_bytes = (Uint32)this->frames->getOutputDataSize();
		this->frame_count = (long int)this->frames->getFrameCount();

		if (this->width <= 0 || this->height <= 0)
			return false;

		return true;
	};

	bool LoadCompressedFile() {
		try {
			//Check if user has access rights to the file
			if (!std::ifstream(this->path).good())
				throw std::runtime_error("No file exists: " + this->path);

			//Ensure file exists and is a valid DICOM file
			OFCondition file_status = this->file_format.loadFile(this->path.c_str());
			if (file_status.bad())
				throw std::runtime_error("Path not valid DICOM image \"DICM\" not present at byte 128\n" + std::string(file_status.text()));

			//Get Dataset representation
			this->dataset = file_format.getDataset();

			//To Decompress Uncomment these lines
			//OFCondition rep_status = this->dataset->chooseRepresentation(this->rep_type, NULL);

			//if (rep_status.bad() || !this->dataset->canWriteXfer(this->rep_type))
			//	throw std::runtime_error("Representation syntax not supported by this DICOM file.");

			//Set DICOM world variables
			if (EC_Normal != dataset->findAndGetLongInt(DCM_Columns, this->width))
				throw std::runtime_error("Could not get Columns tag from: " + this->path);

			if (EC_Normal != dataset->findAndGetLongInt(DCM_Rows, this->height))
				throw std::runtime_error("Could not get Rows tag from: " + this->path);

			if (EC_Normal != dataset->findAndGetLongInt(DCM_SamplesPerPixel, this->samples_per_pixel))
				throw std::runtime_error("Could not get SamplesPerPixel tag from: " + this->path);

			if (EC_Normal != dataset->findAndGetLongInt(DCM_BitsAllocated, this->bits_allocated))
				throw std::runtime_error("Could not get BitsAllocated tag from: " + this->path);

			if (EC_Normal != dataset->findAndGetLongInt(DCM_NumberOfFrames, this->frame_count))
				throw std::runtime_error("Could not get NumberOfFrames tag from: " + this->path);

			if (EC_Normal != dataset->findAndGetElement(DCM_PixelData, this->pixel_data))
				throw std::runtime_error("Could not get pixel data from: " + this->path);

			//Get number of bytes contained for each frame
			this->pixel_data->getUncompressedFrameSize(dataset, frame_bytes);
			return true;
		}
		catch (std::exception err) {
			//Catch any exceptions
			std::cerr << err.what() << std::endl;
			return false;
		}
	};

	//TODO Fix decompression (Currently only works with certain versions of DCMTK/OSX)
	void DecompressDCM(std::string in, std::string out) {
		DcmFileFormat ff;

		//Check for access rights
		if (ff.loadFile(in.c_str()).good())
		{
			//Get dataset
			DcmDataset * ds = ff.getDataset();

			//Force all of data including all frames into memory
			ff.loadAllDataIntoMemory();

			//Convert representation to the most common EXS_LittleEndianExplicit
			OFCondition a = ds->chooseRepresentation(EXS_LittleEndianExplicit, NULL);

			//Ensure the dataset is compatible with the representation
			if (ds->canWriteXfer(EXS_LittleEndianExplicit))
			{
				DcmElement * el = NULL;

				if (EC_Normal == ds->findAndGetElement(DCM_PixelData, el))
				{
					//Write all information to DICOM uncompressed
					ds->saveFile(out.c_str(), EXS_LittleEndianExplicit);
				}
			}
		}
	};

	//Return original representation of a dataset
	OFString GetXfer() {
		OFString xferUID;

		try {
			if (this->file_format.getMetaInfo()->findAndGetOFString(DCM_TransferSyntaxUID, xferUID).bad())
				throw std::runtime_error("Could not get transfer syntax UID of DICOM file: " + this->path);

			DcmXfer xfer(xferUID.c_str());
			E_TransferSyntax xferSyn = xfer.getXfer();
		}
		catch (std::exception err) {
			std::cerr << err.what() << std::endl;
		}

		return xferUID;
	};

	//Return uncompressed data from DICOM file without decompressing entire dataset in memory
	bool GetCompressedFrame(cv::Mat& in, int index) {
		//Attempt to handle multple bit allocations
		try {
			if (index < 0 || index > this->frame_count - 1)
				throw std::runtime_error("Invalid index for DICOM file in GetFrame: " + std::to_string(index));

			//Calculate frame position
			Uint32 frame_offset = (Uint32)index;

			//Store status for both 8 and 16 bit files
			//Only support 8|16
			OFCondition status;

			//Create empty matic incase any errors are thrown
			cv::Mat result(this->height, this->width, CV_MAKETYPE(this->bits_allocated, this->samples_per_pixel));

			if (this->bits_allocated == 8) {
				//Allocate buffer memory
				Uint8 * buffer = new Uint8[int(this->frame_bytes)];
				status = UncompressFrame(buffer, frame_offset);
				//Create matrix and later clone it so that buffer memory doesn't leak
				//TODO: Don't clone the matrix, only have one reference to the created matrix
				result = cv::Mat(this->height, this->width, CV_MAKETYPE(this->bits_allocated, this->samples_per_pixel), (long *)buffer);
				//Clone result matrix for now until buffer issue is resolved
				in = result.clone();
				//Deallocate buffer memory
				free(buffer);
			}
			else if (this->bits_allocated == 16) {
				Uint16 * buffer = new Uint16[int(this->frame_bytes)];
				status = UncompressFrame(buffer, frame_offset);
				result = cv::Mat(this->height, this->width, CV_MAKETYPE(this->bits_allocated, this->samples_per_pixel), (long *)buffer);
				in = result.clone();
				free(buffer);
			}
			else {
				throw std::runtime_error("DICOM file has an unsupported number of bits allocated: " + std::to_string(this->bits_allocated));
			}

			if (status.bad())
				throw std::runtime_error("Couldn't uncompress frame from: " + this->path + " for frame " + std::to_string(index));

			this->frame_index = index;

			//Convert RGB to BGR by swapping the channels
			if (this->samples_per_pixel == 3) {
				std::vector<cv::Mat> channels_bgr;
				cv::split(in, channels_bgr);
				std::vector<cv::Mat> channels_rgb = { channels_bgr[2], channels_bgr[1], channels_bgr[0] };
				cv::merge(channels_rgb, in);
			}
		}
		catch (std::exception err) {
			//TODO: Stop program execution
			std::cerr << err.what() << std::endl;
			return false;
		}

		return true;
	};

	template<typename T>
	OFCondition UncompressFrame(T& buffer, Uint32 frame_offset) {
		//Get uncompressed representation of frame without storing whole dicom file in memory
		//Takes longer but is more memory efficiant
		return this->pixel_data->getUncompressedFrame(this->dataset, frame_offset,
			StartFragment(), buffer, this->frame_bytes, this->decompressed_color_model, this->cache);
	};

	//For getting uncompressed frames, resets position to get next frames from
	//TODO: research better methods
	Uint32& StartFragment() {
		this->start_fragment = 0;
		return this->start_fragment;
	};

	//TODO: Fix to work cross platform
	bool GetDecompressedFrame(cv::Mat& in, int index) {

		try {
			if (index < 0 || index > this->frame_count - 1)
				throw std::runtime_error("Invalid index for DICOM file in GetFrame: " + std::to_string(index));

			//Get pixel data from dicom image at location
			Uint8 *pixel_data = (Uint8 *)this->frames->getOutputData(this->bits_allocated, index);
			cv::Mat result(this->height, this->width, CV_MAKETYPE(this->bits_allocated, this->samples_per_pixel), (long *)pixel_data);

			in = result.clone();
			this->frame_index = index;

			//Convert RGB to BGR
			if (this->samples_per_pixel == 3) {
				std::vector<cv::Mat> channels_bgr;
				cv::split(in, channels_bgr);
				std::vector<cv::Mat> channels_rgb = { channels_bgr[2], channels_bgr[1], channels_bgr[0] };
				cv::merge(channels_rgb, in);
			}
		}
		catch (std::exception err) {
			std::cerr << err.what() << std::endl;
			return false;
		}

		return true;
	};

	int GetWidth() {
		return this->width;
	};

	int GetHeight() {
		return this->height;
	};

	int GetSamplesPerPixel() {
		return this->samples_per_pixel;
	};

	int GetBitsAllocated() {
		return this->bits_allocated;
	};

	int GetFrameCount() {
		return this->frame_count;
	};

	void SetPos(int index = 0) {
		this->frame_index = index;
	}

	int GetPos() {
		return this->frame_index;
	}
private:
	std::string path;
	int frame_index = 0;
	bool compressed;
	long int width, height, samples_per_pixel = 3, bits_allocated, frame_count;
	Uint32 start_fragment = 0, frame_bytes = 0;
	E_TransferSyntax rep_type;
	DcmFileFormat file_format;
	DcmDataset *dataset;
	DcmElement * pixel_data = NULL;
	OFString decompressed_color_model = NULL;
	DcmFileCache *cache = NULL;
	DicomImage * frames;
};