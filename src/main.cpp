#include <string>
#include "BlockMatching.h"
#include "opencv.h"

int main()
{
	std::string srcPath;

#ifdef SOURCE_CODE_LOCATION 
	srcPath = SOURCE_CODE_LOCATION;
#endif

	BlockMatching frames(srcPath + "/input.avi", true);
	frames.sequentialBlockMatch(50);

	return 0;
}
