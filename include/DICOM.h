#include <string>

using namespace std;

class DICOM
{
public:
  DICOM(std::string path);
  void setFilePath(std::string path);

private:
  string filePath;
};