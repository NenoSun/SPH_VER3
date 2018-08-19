// This file is original

#include "OBJGenerator.h"

OBJGenerator::OBJGenerator(std::string filePath) {
	file.open(filePath + std::to_string(i) + ".obj");
	i = 0;
	path = filePath;
}

void OBJGenerator::writeVertice(float x, float y, float z) {
	file << "v " << std::to_string(x) << " " << std::to_string(y) << " " << std::to_string(z) << std::endl;
}

void OBJGenerator::finishThisObject(){
	file.close();
	i++;
	file.open(path + std::to_string(i) + ".obj");
}

void::OBJGenerator::writeTriangle(int count){
	int c = 3 * count + 1;
	file << "f " << std::to_string(c) << " " << std::to_string(c + 1) << " " << std::to_string(c + 2) << std::endl;
}

OBJGenerator::~OBJGenerator() {
	file.close();
}