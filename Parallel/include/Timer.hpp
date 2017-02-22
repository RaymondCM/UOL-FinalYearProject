#include <iostream>
#include <chrono>

class Timer {
public:
	void start() {
		startTime = std::chrono::high_resolution_clock::now();
	};

	long long int end() {
		endTime = std::chrono::high_resolution_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
		return elapsed;
	};

	void print() {
		std::cout << "Elapsed Time [ns]: " << elapsed << std::endl;
	};
private:
	std::chrono::steady_clock::time_point startTime, endTime;
	long long int elapsed;
};