#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

//TODO: Add exception handling
class Capture {
public:
	Capture(std::string device) {
		this->vc = cv::VideoCapture(device);
		this->w = vc.get(cv::CAP_PROP_FRAME_WIDTH);
		this->h = vc.get(cv::CAP_PROP_FRAME_HEIGHT);
	};

	cv::Mat& operator>> (cv::Mat& in)
	{
		this->vc >> in;
		this->frameIndex++;
		return in;
	};

	void reset() {
		this->vc.set(cv::CAP_PROP_POS_AVI_RATIO, 0);
		this->frameIndex = 0;
	};

	bool isOpened() { return this->vc.isOpened(); };

	int width() { return this->w; };

	int height() { return this->h; };
	
	int pos() { return this->frameIndex; };

private:
	cv::VideoCapture vc;
	int w, h, frameIndex = 0;
};