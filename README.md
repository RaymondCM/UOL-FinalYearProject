# ParallelCardiacImaging

## Dependencies
### imebra 
For OSX download the [C++ Source](https://imebra.com/get-it/) and run the following commands to install under `usr/include`, replace `imebra_lib` with the source location. For other platforms instructions can be found on imebra [docs](https://imebra.readthedocs.io/en/stable/compiling_imebra.html)

```
mkdir libOSX 
cd libOSX/
cmake imebra_lib/library/
cmake --build .
make
sudo make install
```