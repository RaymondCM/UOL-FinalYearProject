#include <string>
#include <vector>
#include "opencv.h"

class DICOM
{
public:
  DICOM(std::string);
  DICOM(std::string, bool = false);
  ~DICOM();
  void openCapture();
  cv::Mat getFrameFromCaptured(int);
  void playFrames(u_int, u_int, u_int = 20);
  size_t getFrameCount(){return this->frameCount;}
  int getColsOfCaptured(int n = 0) {return this->getFrameFromCaptured(n).cols;}
  int getRowsOfCaptured(int n = 0) {return this->getFrameFromCaptured(n).rows;}
  bool isRectWithinBounds(cv::Rect, int, int);
  void exhastiveBlockMatch(int = 40);
  void captureFrames();
private:
  std::string filePath;
  std::vector<cv::Mat> frames;
  cv::VideoCapture capture;
  bool isVerbose;
  u_int frameCount = 0, width, height;
  void print(std::string);
};