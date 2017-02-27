#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

namespace BlockMatching {
	inline float square(float x) {
		return x * x;
	}

	inline float euclideanDistance(int x2, int x1, int y2, int y1) {
		return sqrt((float)(square(x2 - x1) + square(y2 - y1)));
	}

	void FullExhastive(cv::Mat& curr, cv::Mat& ref, cv::Point* &motionVectors, int blockSize, int width, int height, int wB, int hB) {
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