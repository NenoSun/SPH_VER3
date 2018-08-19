// This file is original

#pragma once
#include<stdio.h>
#include<fstream>
#include<string>

class OBJGenerator {
private:
	std::ofstream file;
	int i;
	std::string path;

public:
	OBJGenerator(std::string);
	~OBJGenerator();

	void writeVertice(float x, float y, float z);
	void finishThisObject();
	void writeTriangle(int);
};