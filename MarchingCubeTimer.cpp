#pragma once
#include "MarchingCubeTimer.h"

MarchingCubeTimer::MarchingCubeTimer(std::string path) {
	count = 0;
	file.open(path);
}

void MarchingCubeTimer::start() {
	QueryPerformanceCounter(&startTime);
}

void MarchingCubeTimer::end() {
	QueryPerformanceCounter(&endTime);
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	__int64 diff = endTime.QuadPart - startTime.QuadPart;
	file << count << " " << (float)diff / (float)frequency.QuadPart << std::endl;
	count++;
}

