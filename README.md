# ParallelCardiacImaging

## Dependencies		
### imebra 		
For OSX/Linux download the [C++ Source](https://imebra.com/get-it/) and run the following commands to install under `usr/include`, replace `imebra_source` with the source location. For other platforms instructions can be found on imebra [docs](https://imebra.readthedocs.io/en/stable/compiling_imebra.html)	
		
```
mkdir imebra_bin
cd imebra_bin/
cmake imebra_source/library/
cmake --build .
make
sudo make install
```

### OpenCV
For OSX/Linux download the [C++ Source](http://opencv.org/downloads.html) and run the following commands to install under `usr/local`, replace `opencv_downloaded_source` with the source location. For other platforms instructions can be found on imebra [docs](http://docs.opencv.org/2.4/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html)	
 
```
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
cd opencv_downloaded_source
mkdir opencv_bin
cd opencv_bin
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
```

## Installation
From project root run. Or use [CMake for VSCode](https://marketplace.visualstudio.com/items?itemName=vector-of-bool.cmake-tools).

```
cmake -H./ -B./build -C./build/CMakeTools/InitializeCache.cmake "-GUnix Makefiles"
cd build/
cmake ..
make
```

## Execute
From project root.

```
cd build/
make project
./project
```