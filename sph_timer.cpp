// This file sources from https://github.com/finallyjustice/sphfluid


#include "Configuration.h"
#ifdef TIMER
#include "sph_timer.h"

Timer::Timer(const std::string filePath) {
	lastTime = 0;
	startTime = 0;
	isStarted = false;
	file.open(filePath);
	frameCount = 0;
}

void Timer::update() {
	if (!isStarted)
		return;
	int currentTime = GetTickCount();
	float interval = (currentTime - lastTime) / 1000.0;
	lastTime = currentTime;
	records.push_back(interval);
	frameCount++;
	std::cout << "Interval: " << interval << std::endl;
	file << frameCount << " " << interval << std::endl;
}

void Timer::end() {
	if (!isStarted)
		return;
	int currentTime = GetTickCount();
	float interval = (currentTime - lastTime) / 1000.0;
	lastTime = currentTime;
	records.push_back(interval);
	for(int i = 0; i < records.size(); i++)
		std::cout << "END Interval: " << records[i] << std::endl;
	isStarted = false;
}

void Timer::start() {
	if (!isStarted)
		return;
	startTime = GetTickCount();
	lastTime = startTime;
}

Timer::~Timer() {
	file.close();
}

#endif
