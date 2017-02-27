#include <iostream>
#include <vector>
#include <chrono>

#define NANO 1000000000.0

class Timer {
public:
	Timer(int maxSamp = 50) {
		this->maxSamples = maxSamp;
	};

	void tic() {
		this->startTime = std::chrono::high_resolution_clock::now();
	};

	void toc() {
		this->timestamps.push_back(this->getElapsed());
		if (this->timestamps.size() > maxSamples) {
			this->timestamps.erase(this->timestamps.begin(), this->timestamps.begin() + this->timestamps.size() - maxSamples);
		}
	};

	long long int stop() {
		timestamps.clear();
		return this->getElapsed();
	};

	float getFPSFromElapsed() {
		float sum = elapsedSum();
		return sum <= 0 ? 0 : this->timestamps.size() / sum;
	};

	float elapsedSum() {
		long long int sum = 0;

		for (std::vector<long long int>::iterator it = this->timestamps.begin(); it != this->timestamps.end(); ++it)
			sum += *it;

		return (float) sum / NANO;
	};

	long long int getElapsed() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
	};
private:
	std::chrono::steady_clock::time_point startTime;
	std::vector<long long int> timestamps;
	unsigned int maxSamples;
};