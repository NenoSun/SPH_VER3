// This file is mostly original with some references from the following github repository.
// https://github.com/finallyjustice/sphfluid

#ifndef SPHSYSTEM
#define SPHSYSTEM

#include "Type.cuh"
#include "Param.h"
#include <math.h>
#include "source.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include "Configuration.h"
#include "PhysicalEngineTimer.h"
#include "CpuDFSPH.h"


#define PI 3.1415926
#define ZERO 1e-12
#define BOUNDARY 0.001f


#define POLY6_KERNEL(x)  (poly6_coff * pow(h_square - x, 3))
#define POLY6_GRAD_KERNEL(x) a;

#define CUBE_HASH(i,j,k) (k*parameters.cubeCount.x*parameters.cubeCount.y+ j*parameters.cubeCount.x + i)
#define SQUARE(x) (x*x)

using namespace std;


// This is the base class for SPHSystem
class SPHSystem
{
public:
	int sys_running;  // Indicates the runnning status of the system.
	Param parameters;

	// GPU Function
	Particle *hBoundaryParticles, *dBoundaryParticles; // Array of boundary particles. One for the host ( Main memory ) , one for the device ( GPU memory )
	Particle* dParticles, *hParticles;  // Array of fluid particles. One for the host ( Main memory ) , one for the device ( GPU memory )
	//Param *dParam, *hParam; // Parameter object. 
	Param *dParam; // Parameter object. 
	uint* dParticleIndex, *dCellIndex; // For uniform grid sorting use
	uint* dBoundaryParticleIndex, *dBoundaryCellIndex; // For uniform grid sorting use
	uint* dStart, *dEnd; // For uniform grid sorting use
	uint* dBoundaryStart, *dBoundaryEnd; // For uniform grid sorting use

#ifdef CPU_DF
	uint* hParticleIndex, *hCellIndex; // For uniform grid sorting use
	uint* hBoundaryParticleIndex, *hBoundaryCellIndex; // For uniform grid sorting use
	uint* hStart, *hEnd; // For uniform grid sorting use
	uint* hBoundaryStart, *hBoundaryEnd; // For uniform grid sorting use
#endif


	// Just for test
	float isDFSPHReady;

public:
	SPHSystem();
	~SPHSystem();

	// Copy the system's parameters to a param object
	void PassParamsTo_hPram(Param *param);

	// Generate inital particles to the system
	void generateParticles();

	// Add one particular particle to the system
	void addParticle(Float3 pos, Float3 vel);
	void addBoundaryParticle(Float3 pos, Float3 vel, bool isObject);
	void addSphericalObject(Float3 center, float radius);
	void addObject(string url, Float3 pos);

	void animation();

	void computeThreadsAndBlocks(int maxThreadsPerBlock, int num_particles);


public:
	// Marching cube part
	cube *hCubes, *dCubes;
	Float3 *hTriangles, *dTriangles;
	Float3 *hNorms, *dNorms;
	void MarchingCubeSetUp();
};

#endif