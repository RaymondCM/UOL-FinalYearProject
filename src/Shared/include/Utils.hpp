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

	cv::Point down_point(0, 0), up_point(0, 0);
	bool down = false, up = false, roi_done = false;

	void ROIMouseCallback(int event, int x, int y, int flags, void* userdata)
	{
		switch (event) {
		case(cv::EVENT_LBUTTONDOWN):
			if (!down) {
				up = false;
				down = true;
				down_point = cv::Point(x, y);
			}
			break;
		case(cv::EVENT_LBUTTONUP):
			if(!up)
			{
				down = false;
				up = true;
				up_point = cv::Point(x, y);
			}
			break;
		case(cv::EVENT_MOUSEMOVE):
			if(down && !up)
				up_point = cv::Point(x, y);
		default:
			break;
		}
	}

	cv::Rect WaitForROI(std::string winname, cv::Mat image)
	{
		cv::Mat original = image.clone();

		while(!roi_done)
		{
			image = original.clone();

			if(down_point != up_point)
				cv::rectangle(image, down_point, up_point, cv::Scalar(255));

			cv::imshow(winname, image);
			if(std::tolower(static_cast<char>(cv::waitKey(1))) == 'y')
			{
				roi_done = true;
				int round_to = 10;
				down_point.x = ((down_point.x + round_to / 2) / round_to) * round_to;
				down_point.y = ((down_point.y + round_to / 2) / round_to) * round_to;
				up_point.x = ((up_point.x + round_to / 2) / round_to) * round_to;
				up_point.y = ((up_point.y + round_to / 2) / round_to) * round_to;
			}
		}

		return cv::Rect(down_point, up_point);
	}
}