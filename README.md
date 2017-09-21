# ParallelCardiacImaging

Presented is a GPU implementation of a Full Search Block Matching Motion
Estimation algorithm using sum of absolute difference error criterion and a novel criterion to find
optimal block displacement, and to estimate the heart rate of apical four chamber
view ultrasound images. With the rapid growth of GPU processing power in recent
years an effective deployment medium for parallelisation of traditionally serial
algorithms exists. This project will attempt to evaluate the feasibility of using heterogeneous
systems for motion estimation by comparing algorithm execution times
and accuracy on the CPU and GPU. This thesis concludes by showing the GPU implementation
is sufficiently fast enough for real-time processing and can shorten
computation time of accurate heart rate estimates by a factor of 450 times ([Link](https://github.com/RaymondKirk/UOL-FinalYearProject/blob/master/uol-finalyearthesis.pdf)).

## Dependencies		
### DCMTK 		
Instructions below for Ubuntu 14.04, the stable release has [compiler issues](http://forum.dcmtk.org/viewtopic.php?f=1&t=4235) so a snapshot from GitHub is used. 

```
wget http://git.dcmtk.org/?p=dcmtk.git;a=snapshot;h=681e3182ccfe3873b95824b07a4565c8b54a8a18;sf=tgz
tar xzvf index.html\?p=dcmtk.git\;a=snapshot\;h=681e3182ccfe3873b95824b07a4565c8b54a8a18\;sf=tgz
cmake-gui & #Configure and Generate make files to build_folder
cd build_folder
cmake --build .
make #(Optional?)
sudo make install
```

### OpenCV
#### Windows
Download the [OpenCV](https://github.com/opencv/opencv) source and build with cmake (See instructions below). Build the binaries for debug and release to "C:/OpenCV/build" with OPENCL. Build INSTALL to produce system headers and dynamic linked libraries. Add a system environment variable to the installed OpenCV_DIR (C:\OpenCV\build\install\x86\vc15 for Visual Studio 15 2017). Finally add a variable to system PATH (%OpenCV_DIR%/bin). See OpenCV [documentation](http://docs.opencv.org/2.4/doc/tutorials/introduction/windows_visual_studio_Opencv/windows_visual_studio_Opencv.html) for guidelines and resources.

To build use CMake-GUI with these options WITH_CUDA=0, WITH_OPENCL=1 and BUILD_opencv_java=0. Optionally add both python interpreters.

```
cmake -H./ -BC:/OpenCV/build/ -G "Visual Studio 15 2017" -DWITH_CUDA=0 -DWITH_OPENCL=1 -DBUILD_opencv_java=0 #USE CMake-GUI
```

To set environment variables to installed location.

```
setx -m OpenCV_DIR C:\OpenCV\build\install\x86\vc15
setx -m PATH=%PATH%;%OpenCV_DIR%\bin
```

When compiling add ```-DCMAKE_PREFIX_PATH=C:\OpenCV\build\install``` for ProjectY3 target when OpenCV cannot be found.

#### Linux (Ubuntu)
For OSX/Linux download the [C++ Source](http://opencv.org/downloads.html) and run the following commands to install under `usr/local`, replace `opencv_downloaded_source` with the source location. For other platforms instructions can be found on imebra [docs](http://docs.opencv.org/2.4/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html)	
 
```
cd opencv_downloaded_source
mkdir opencv_bin
cd opencv_bin
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
```

OpenCV dependencies (ubuntu) listed below.

``` 
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```
### OpenCL
#### OSX
[C++ bindings](https://www.khronos.org/registry/OpenCL/api/2.1/cl.hpp) not included by default with OpenCL, download and include them.

## Compile
On ubuntu from project root run ```cmake -H./ -B./build``` or use [CMake for VSCode](https://marketplace.visualstudio.com/items?itemName=vector-of-bool.cmake-tools).

To execute the program type the following code from the root of the file structure.

```
cd build/
make clean #OPTIONAL
make ProjectY3
./ProjectY3
```
