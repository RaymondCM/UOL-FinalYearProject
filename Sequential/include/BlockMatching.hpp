#include <string>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

namespace BlockMatching {
	inline float square(float x) {
		return x * x;
	}

	inline int AbsoluteDifference(int a, int b) {
		return a < b ? b - a : a - b;
	}
	inline float euclideanDistance(int x2, int x1, int y2, int y1) {
		return sqrt((float)(square(x2 - x1) + square(y2 - y1)));
	}

	inline int MatrixSum(cv::Mat img, cv::Point p, int blockSize) {
		//Force value to be absolute to force the equality|x+y|<=|x|+|y| because xy=|x||y|=|xy|
		return cv::sum(cv::abs(img(cv::Rect(p.x, p.y, blockSize, blockSize))))[0];
	}

	inline bool IsInBounds(int x, int y, int width, int height, int bSize) {
		return y >= 0 && y < height - bSize && x >= 0 && x < width - bSize;
	}

	cv::Vec3i ClosestNeighbour(cv::Mat prev, cv::Point currPoint, const int sWindow, int width, int height, int blockSize) {
		for (int row = -sWindow; row < sWindow; row += sWindow) {
			for (int col = -sWindow; col < sWindow; col += sWindow) {
				cv::Point refPoint(currPoint.x + row, currPoint.y + col);
				//Check if the block is within the bounds to avoid incorrect values
				if (IsInBounds(refPoint.x, refPoint.y, width, height, blockSize)) {
					return cv::Vec3i(row, col, MatrixSum(prev, refPoint, blockSize));
				}
			}
		}

		cv::Point refPoint(currPoint.x, currPoint.y);
		return cv::Vec3i(0, 0, MatrixSum(prev, refPoint, blockSize));
	}

	void FullExhastive(cv::Mat& curr, cv::Mat& ref, cv::Vec4f* &motionVectors, int blockSize, int width, int height, int wB, int hB) {
		//Loop over all possible blocks in frame
		for (int x = 0; x < wB; x++) {
			for (int y = 0; y < hB; y++) {
				//Reference point on current frame that will be searched for in the previous frame
				const cv::Point currPoint(x * blockSize, y * blockSize);
				int idx = x + y * wB;

				int current_err = MatrixSum(curr, currPoint, blockSize);

				const int sWindow = blockSize;
				cv::Vec3i closest = ClosestNeighbour(ref, currPoint, sWindow, width, height, blockSize);
				//int ref_err = closest[2];

				float distanceToBlock = FLT_MAX;
				int bestErr = INT_MAX, err;

				//Loop over all possible blocks within each macroblock
				for (int row = closest[0]; row < sWindow; row++) {
					for (int col = closest[1]; col < sWindow; col++) {
						//Refererence a block to search on the previous frame
						cv::Point refPoint(currPoint.x + row, currPoint.y + col);

						//Check if it lays within the bounds of the capture
						if (IsInBounds(refPoint.x, refPoint.y, width, height, blockSize)) {
							//Calculate SSD (Sum of square differences)

							int ref_err = MatrixSum(ref, refPoint, blockSize);
							err = AbsoluteDifference(current_err, ref_err);

							//Take the lowest error, closeness is preffered.
							float newDistance = euclideanDistance(refPoint.x, currPoint.x, refPoint.y, currPoint.y);

							//Write buffer with the lowest error
							if (err < bestErr) {
								bestErr = err;
								distanceToBlock = newDistance;
								float p0x = currPoint.x, p0y = currPoint.y - sqrt((float)(square(refPoint.x - p0x) + square(refPoint.y - currPoint.y)));
								float angle = (2 * atan2(refPoint.y - p0y, refPoint.x - p0x)) * 180 / M_PI;
								motionVectors[idx] = cv::Vec4f(refPoint.x, refPoint.y, angle, distanceToBlock);
							}
							else if (err == bestErr && newDistance <= distanceToBlock) {
								distanceToBlock = newDistance;
								float p0x = currPoint.x, p0y = currPoint.y - sqrt((float)(square(refPoint.x - p0x) + square(refPoint.y - currPoint.y)));
								float angle = (2 * atan2(refPoint.y - p0y, refPoint.x - p0x)) * 180 / M_PI;
								motionVectors[idx] = cv::Vec4f(refPoint.x, refPoint.y, angle, distanceToBlock);
							}
						}
					}
				}
			}
		}
	}

	void NaiveFullExhastive(cv::Mat& curr, cv::Mat& ref, cv::Point* &motionVectors, int blockSize, int width, int height, int wB, int hB) {
		//Loop over all possible blocks in frame
		for (int x = 0; x < wB; x++) {
			for (int y = 0; y < hB; y++) {
				//Reference point on current frame that will be searched for in the previous frame
				const cv::Point currPoint(x * blockSize, y * blockSize);
				int idx = x + y * wB;

				cv::Rect middle(currPoint.x, currPoint.y, blockSize, blockSize);
				const int sWindow = blockSize;
				float distanceToBlock = FLT_MAX;
				float bestErr = FLT_MAX, err;

				//Loop over all possible blocks within each macroblock
				for (int row = -sWindow; row < sWindow; row++) {
					for (int col = -sWindow; col < sWindow; col++) {
						//Refererence a block to search on the previous frame
						cv::Point refPoint(currPoint.x + row, currPoint.y + col);

						//Check if it lays within the bounds of the capture
						if (refPoint.y >= 0 && refPoint.y < height - blockSize && refPoint.x >= 0 && refPoint.x < width - blockSize) {
							//Calculate SSD (Sum of square differences)
							cv::Mat diff = cv::abs(curr(middle) - ref(cv::Rect(refPoint.x, refPoint.y, blockSize, blockSize)));
							err = cv::sum(diff)[0];

							//Take the lowest error, closeness is preffered.
							float newDistance = euclideanDistance(refPoint.x, currPoint.x, refPoint.y, currPoint.y);

							//Write buffer with the lowest error
							if (err < bestErr) {
								bestErr = err;
								distanceToBlock = newDistance;
								motionVectors[idx] = refPoint;
							}
							else if (err == bestErr && newDistance <= distanceToBlock) {
								distanceToBlock = newDistance;
								motionVectors[idx] = refPoint;
							}
						}
					}
				}
			}
		}
	}
}