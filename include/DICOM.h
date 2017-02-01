#include <string>
#include <vector>
#include "opencv.h"

class DICOM
{
public:
  DICOM(std::string);
  DICOM(std::string, bool);
  ~DICOM();
  cv::Mat getFrame(int);
  void playFrames(u_int, u_int, u_int = 20);
  size_t getFrameCount(){return this->frameCount;};
private:
  std::string filePath;
  std::vector<cv::Mat> frames;
  bool isVerbose;
  size_t frameCount = 0;
  void captureVideoFrames();
  void print(std::string);
};