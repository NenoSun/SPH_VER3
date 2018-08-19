// This file is original

#include "Configuration.h"
#ifdef TIMER
#ifndef __PHYSICAL_ENGINE_TIMER
#define __PHYSICAL_ENGINE_TIMER
#include <fstream>
#include <Windows.h>

class PhysicalEngineTimer {
public:
	int startTime;
	int endTime;
	int frameCount;
	std::ofstream file;

	void init(const std::string);
	PhysicalEngineTimer(const std::string filePath);
	~PhysicalEngineTimer();
	void start();
	void end();
};

#endif
#endif