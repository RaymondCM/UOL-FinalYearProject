#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

//TODO: Add exception handling
class Capture {
public:
	Capture(std::string device) {
		this->vc = cv::VideoCapture(device);
		this->frame_count = vc.get(cv::CAP_PROP_FRAME_COUNT);
		this->width = vc.get(cv::CAP_PROP_FRAME_WIDTH);
		this->height = vc.get(cv::CAP_PROP_FRAME_HEIGHT);
	};

	cv::Mat& operator>> (cv::Mat& in)
	{
		this->vc >> in;
		this->frame_index++;
		return in;
	};

	void Reset() {
		this->vc.set(cv::CAP_PROP_POS_AVI_RATIO, 0);
		this->frame_index = 0;
	};

	bool IsOpened() { return this->vc.isOpened(); };

	int GetWidth() { return this->width; };

	int GetHeight() { return this->height; };
	
	void SetPos(int index = 0) { this->frame_index = index; };
	
	int GetPos() { return this->frame_index; };

	int GetFrameCount() {
		return this->frame_count;
	};

	bool isLastFrame() {
		return this->frame_count - 1 == this->frame_index;
	}
private:
	cv::VideoCapture vc;
	int width, height, frame_index = 0, frame_count;
};