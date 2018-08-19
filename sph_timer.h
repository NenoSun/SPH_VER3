// This file sources from https://github.com/finallyjustice/sphfluid

#include "Configuration.h"
#ifdef TIMER
#include <vector>
#ifndef __SPHTIMER_H__
#define __SPHTIMER_H__
#include <windows.h>
#include <iostream>
#include <fstream>

class Timer
{
public:
	int startTime;
	int lastTime;
	std::vector<int> records;
	bool isStarted;
	std::ofstream file;
	int frameCount;


	Timer(const std::string a);
	~Timer();
	void start();
	void update();
	void end();
};

#endif
#endif
