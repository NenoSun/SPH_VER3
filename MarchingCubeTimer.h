// This file is original

#pragma once
#include <Windows.h>
#include <fstream>
class MarchingCubeTimer {
public:
	LARGE_INTEGER startTime;
public:
	LARGE_INTEGER endTime;
	int count;
	std::ofstream file;

	MarchingCubeTimer(std::string path);

	void start();

	void end();

};