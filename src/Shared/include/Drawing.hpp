#include <opencv2/opencv.hpp>

namespace Draw {
	inline float square(float x) {
		return x * x;
	}

	inline float euclideanDistance(int x2, int x1, int y2, int y1) {
		return sqrt((float)(square(x2 - x1) + square(y2 - y1)));
	}

	inline float euclideanDistance(cv::Point a, cv::Point b) {
		return sqrt((float)(square(b.x - a.x) + square(b.y - a.y)));
	}

	template<typename T>
	T AlphaBlend(T a, T b, float alpha = 0.5) {
		float beta = 1 - alpha;
		return (alpha * a) + (beta * b);
	}

	//Based on https://link.springer.com/book/10.1007/b138805
	cv::Scalar HSVToBGR(float h, float s, float v) //H[0, 360], S[0, 1], V[0, 1]
	{
		cv::Scalar out;
		double P, Q, T, fr;

		(h == 360.) ? (h = 0.) : (h /= 60.);
		fr = h - floor(h);

		P = v*(1. - s);
		Q = v*(1. - s*fr);
		T = v*(1. - s*(1. - fr));

		if (0. <= h && h < 1.)
			out = cv::Scalar(P, T, v, s);
		else if (1. <= h && h < 2.)
			out = cv::Scalar(P, v, Q, s);
		else if (2. <= h && h < 3.)
			out = cv::Scalar(P, v, T, s);
		else if (3. <= h && h < 4.)
			out = cv::Scalar(P, Q, v, s);
		else if (4. <= h && h < 5.)
			out = cv::Scalar(v, P, T, s);
		else if (5. <= h && h < 6.)
			out = cv::Scalar(v, P, Q, s);
		else
			out = cv::Scalar(0, 0, 0, 0);

		out *= 255;

		return out;
	}

	void Arrow(cv::Mat& canvas, cv::Point2f p) {
		cv::Point2f center(canvas.cols / 2, canvas.rows / 2);
		cv::arrowedLine(canvas, center, p, cv::Scalar(0, 0, 255), 5);
	}

	template<typename T>
	void MotionVectors(cv::Mat &canvas, T *& motionVectors, unsigned int wB, unsigned int hB, int blockSize, int stepSize, bool drawGrid = false,
		cv::Scalar rectColour = cv::Scalar(255), cv::Scalar lineColour = cv::Scalar(0, 255, 255)) {
		for (size_t i = 0; i < wB; i++)
		{
			for (size_t j = 0; j < hB; j++)
			{
				//Calculate repective position of motion vector
				int idx = i + j * wB;

				//Offset drawn point to represent middle rather than top left of block
				cv::Point offset(blockSize / 2, blockSize / 2);
				cv::Point pos(i * stepSize, j * stepSize);
				cv::Point mVec(motionVectors[idx].x, motionVectors[idx].y);

				if (drawGrid)
					cv::rectangle(canvas, pos, pos + cv::Point(blockSize, blockSize), rectColour);

				cv::arrowedLine(canvas, pos + offset, mVec + offset, lineColour);
			}
		}
	}

	template<typename T, typename X>
	void MotionVectorHSVAngles(cv::Mat &canvas, T *& motionVectors, X *& motionDetails, unsigned int wB, unsigned int hB, int blockSize, int stepSize,
		int thresh = 1, float min_len = 0.0) {

		cv::Mat mask;
		cv::cvtColor(canvas, mask, cv::COLOR_RGB2GRAY);
		threshold(mask, mask, thresh, 255, 0);

		cv::Mat colour_image = canvas.clone(), dst;
		float max_len = euclideanDistance(cv::Point(0, 0), cv::Point(blockSize, blockSize));

		for (size_t i = 0; i < wB; i++)
		{
			for (size_t j = 0; j < hB; j++)
			{
				//Calculate repective position of motion vector
				int idx = i + j * wB;

				//Offset drawn point to represent middle rather than top left of block
				cv::Point offset(blockSize / 2, blockSize / 2);
				cv::Point pos(i * stepSize, j * stepSize);
				cv::Point mVec(motionVectors[idx].x, motionVectors[idx].y);

				float len = (motionDetails[idx].y / max_len);
				if (len >= min_len) {
					float angle = motionDetails[idx].x;

					cv::rectangle(colour_image, pos, pos + cv::Point(blockSize, blockSize), HSVToBGR(angle, len, 1), CV_FILLED);

					//Attempt to merge existing colours to weight more occuring angle/lengths higher
					/*cv::Scalar hsv = HSVToBGR(angle, len, 1);
					for (int x = pos.x; x < pos.x + blockSize; x++) {
					for (int y = pos.y; y < pos.y + blockSize; y++) {
					if (x > 0 && y > 0 && x < colour_image.cols && y < colour_image.rows) {
					cv::Scalar src_pixel = colour_image.at<cv::Vec<unsigned char, 4>>(cv::Point(x, y));
					cv::Scalar c_pixel = canvas.at<cv::Vec<unsigned char, 4>>(cv::Point(x, y));
					if (src_pixel == c_pixel) {
					colour_image.at<cv::Vec<unsigned char, 4>>(cv::Point(x, y)) = hsv;
					}
					else {
					cv::Scalar blended = AlphaBlend(src_pixel, hsv, 0.2);
					colour_image.at<cv::Vec<unsigned char, 4>>(cv::Point(x, y)) = blended;
					}
					}
					}
					}*/

					//cv::putText(colour_image, std::to_string((int)angle), pos, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.4, cv::Scalar(255, 255, 255));
					//std::cout << "A:" << angle << ", L:" << len << std::endl;
				}
			}
		}

		//Blur image
		cv::boxFilter(colour_image, colour_image, -1, cv::Size(blockSize, blockSize), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);

		for (int i = 0; i < mask.cols; i++) {
			for (int j = 0; j < mask.rows; j++) {
				if (mask.at<uchar>(j, i) > 0) {
					//Simple alpha blending 50/50
					canvas.at<cv::Vec3b>(j, i) = AlphaBlend(canvas.at<cv::Vec3b>(j, i), colour_image.at<cv::Vec3b>(j, i), 0.2);
				}
			}
		}

		//cv::addWeighted(canvas, 0.5, colour_image, 0.5, 0.0, canvas);
	}

	void Text(cv::Mat& canvas, std::string f, std::string bS, std::string sS, std::string processed_fps, std::string rendered_fps, cv::Scalar colour = cv::Scalar(255, 255, 255)) {
		std::string content("Frame " + f + ", Block Size: " + bS + ", Step Size: " + sS + ", Processed FPS: " + processed_fps + ", Rendered FPS: " + rendered_fps);
		cv::putText(canvas, content, cv::Point(0, canvas.size().height - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, colour);
	}
}