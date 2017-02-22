#include <iostream>
#include <chrono>

class Timer {
public:
	void start() {
		startTime = std::chrono::high_resolution_clock::now();
	};

	long long int end() {
		endTime = std::chrono::high_resolution_clock::now();
		long long int elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
		std::cout << "Elapsed Time: " << elapsed << " [ns]" << std::endl;
		return elapsed;
	};

private:
	std::chrono::steady_clock::time_point startTime, endTime;
};