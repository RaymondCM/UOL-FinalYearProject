#include <string>
#include <vector>
#include <algorithm>

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

	int getStepSize(int blockSize) {
		std::vector<int> f;
		getFactors(f, blockSize);
		std::sort(f.begin(), f.end());

		//Return second to last element (discount using blockSize)
		return f.rbegin()[1];
	}

	template<typename T, typename X> //X,Y MAGNITUDE, ANGLE
	cv::Vec4f analyseData(T*& motion_points, X*& motion_info, int size) {
		cv::Point2f average_point, point_sum(0, 0);
		double average_magnitude = 0, magnitude_sum = 0;
		double average_angle = 0, angle_sum = 0;

		for (int i = 0; i < size; ++i) {
			point_sum += cv::Point2f(motion_points[i].x, motion_points[i].y);
			angle_sum += motion_info[i].x;
			magnitude_sum += motion_info[i].y;
		}

		average_point = point_sum / size;
		average_magnitude = magnitude_sum / size;
		average_angle = angle_sum / size;

		//std::cout << "Averages: Point(" << average_point.x << ", " << average_point.y 
		//<< ")\tMagnitude(" << average_magnitude << ")\tAngle(" << average_angle << ")" << std::endl;

		return cv::Vec4f(average_point.x, average_point.y, average_magnitude, average_angle);
	}

	void MouseCallback(int event, int x, int y, int flags, void* userdata)
	{
		switch (event) {
		case(cv::EVENT_LBUTTONDOWN):
			break;
		case(cv::EVENT_RBUTTONDOWN):
			break;
		case(cv::EVENT_MBUTTONDOWN):
			break;
		case(cv::EVENT_MOUSEMOVE):
			break;
		default:
			break;
		}
	}
}