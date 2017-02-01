# ParallelCardiacImaging

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
make ProjectY3
./ProjectY3
```