#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

namespace Util {
	void getFactors(std::vector<int> & factors, int number) {
		factors.push_back(1);
		factors.push_back(number);
		for (int i = 2; i * i <= number; ++i)
		{
			if (number % i == 0)
			{
				factors.push_back(i);
				if (i * i != number) {
					factors.push_back(number / i);
				}
			}
		} 
	}

	std::vector<int> getBlockSizes(int w, int h) {
		std::vector<int> wF, hF, c;
		getFactors(wF, w);
		getFactors(hF, h);
		
		std::sort(wF.begin(), wF.end());
		std::sort(hF.begin(), hF.end());

		std::set_intersection(wF.begin(), wF.end(), hF.begin(), hF.end(), std::back_inserter(c));

		return c;
	}

	template<typename T>
	void drawMotionVectors(cv::Mat &canvas, T *& motionVectors, unsigned int wB, unsigned int hB, int blockSize,
		cv::Scalar rectColour = cv::Scalar(255), cv::Scalar lineColour = cv::Scalar(0, 255, 255)) {
		for (size_t i = 0; i < wB; i++)
		{
			for (size_t j = 0; j < hB; j++)
			{
				//Calculate repective position of motion vector
				int idx = i + j * wB;

				//Offset drawn point to represent middle rather than top left of block
				cv::Point offset(blockSize / 2, blockSize / 2);
				cv::Point pos(i * blockSize, j * blockSize);
				cv::Point mVec(motionVectors[idx].x, motionVectors[idx].y);

				cv::rectangle(canvas, pos, pos + cv::Point(blockSize, blockSize), rectColour);
				cv::arrowedLine(canvas, pos + offset, mVec + offset, lineColour);
			}
		}
	}

	void drawText(cv::Mat& canvas, std::string f, std::string bS, std::string fps, cv::Scalar colour = cv::Scalar(255, 255, 255)) {
		std::string content("Frame " + f + ", Block Size: " + bS + ", FPS: " + fps);
		cv::putText(canvas, content, cv::Point(0, canvas.size().height - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, colour);
	}
}