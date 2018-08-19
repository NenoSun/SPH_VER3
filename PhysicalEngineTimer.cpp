// This file is original

#include "PhysicalEngineTimer.h"
#ifdef TIMER

PhysicalEngineTimer::PhysicalEngineTimer(const std::string filePath) {
	startTime = 0;
	endTime = 0;
	frameCount = 0;
	file.open(filePath);
}

void PhysicalEngineTimer::init(const std::string filePath) {
	startTime = 0;
	endTime = 0;
	frameCount = 0;
	file.open(filePath);
}

PhysicalEngineTimer::~PhysicalEngineTimer() {
	file.close();
}

void PhysicalEngineTimer::start() {
	startTime = GetTickCount();
	
}

void PhysicalEngineTimer::end() {
	endTime = GetTickCount();
	frameCount++;
	file << frameCount << " " << (float)(endTime - startTime) / 1000.0 << std::endl;
}

#endif