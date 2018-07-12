//#pragma once
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


#define PI 3.1415926
#define ZERO 1e-12
#define BOUNDARY 0.001f


#define POLY6_KERNEL(x)  (poly6_coff * pow(h_square - x, 3))
#define POLY6_GRAD_KERNEL(x) a;

#define CUBE_HASH(i,j,k) (k*cubeCount.x*cubeCount.y+ j*cubeCount.x + i)
#define SQUARE(x) (x*x)

using namespace std;


// This is the base class for SPHSystem
class SPHSystem
{
public:
	int sys_running;  // Indicates the runnning status of the system.
	// Particles world parameters
	uint num_max; // Maxmimal particle number, it doesn't work.
	float mass; // The mass of each particle. 

	uint num_particles; // The current particle number
	uint num_boundary_p; // The boundary particle number

	// Hash
	Float3 worldSize;  // The Box size
	Uint3 gridSize; // The grid size
	uint grid_num; // How many grids there are
	float cell_size; // How many cells there are

	// Function parameters
	float h; // Finite support radius
	float h_square; // The square of finite support radius, reduce calculation complexity
	float rest_density; // Rest density
	float l_threshold; // 
	float timeStep; // timestep between each frame. The timestep is not fixed here. It changes based on the distribution of fluid particles.


	// Coefficients of different formular
	float gas_stiffness; 
	float vicosity_coff;
	float surf_tens_coff;
	float wall_damping;


	// The coefficient of different kernels (Poly6 Kernel, Spiky Kernel and Viscosity Kernel)
	float poly6_coff;
	float grad_spiky_coff;
	float lplc_visco_coff;
	float grad_poly6;
	float lplc_poly6;
	float spline_coff;
	float grad_spline_coff;
	float cohesion_coff;
	float cohesion_term;
	float Adhesion_coff;


	// Temporary parameters
	uint hash;
	Uint3 neightborPos;
	Float3 Gravity;
	Float3 sim_ratio;


	// GPU Function
	int BLOCK; // Block number
	int THREAD; // Thread number
	Particle *hBoundaryParticles, *dBoundaryParticles; // Array of boundary particles. One for the host ( Main memory ) , one for the device ( GPU memory )
	Particle* dParticles, *hParticles;  // Array of fluid particles. One for the host ( Main memory ) , one for the device ( GPU memory )
	Param *dParam, *hParam; // Parameter object. 
	uint* dParticleIndex, *dCellIndex; // For uniform grid sorting use
	uint* dBoundaryParticleIndex, *dBoundaryCellIndex; // For uniform grid sorting use
	uint* dStart, *dEnd; // For uniform grid sorting use
	uint* dBoundaryStart, *dBoundaryEnd; // For uniform grid sorting use


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
	float isovalue; // Determine whether a vertex should be inside or outside.
	cube *hCubes, *dCubes; 
	float cubeSize;
	uint cubePerAxies;
	Uint3 cubeCount;
	uint cube_num;
	Float3 *hTriangles, *dTriangles;
	Float3 *hNorms, *dNorms;



	void MarchingCubeSetUp();
	void MarchingCubeRun();
	void MarchingCubeGenerateFirstCubes();

	float calDistance(Float3 p1, Float3 p2);
};

#endif