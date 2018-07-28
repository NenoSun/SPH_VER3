#pragma once
#include <math.h>
#include "Type.cuh"
#include "stdio.h"
#include "Configuration.h"

#define PI 3.1415926

class Param
{
public:
	float mass;

	float h ;
	float h_square;
	float rest_density;
	float l_threshold ;
	float timeStep;

	float gas_stiffness;
	float vicosity_coff;
	float surf_tens_coff;
	float wall_damping;

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

	float self_dens;

	float SpeedOfSound;
	float B;

	Float3 worldSize;
	Uint3 gridSize;
	int cells_total;
	uint num_particles;
	uint num_boundary_particles;
	uint max_num_particles;

	float avg_grad_dens;
	float avg_dens;
	float radius;

#ifdef RENDER_MESH
	// Marching Cube Parameters
	uint cubePerAxies;
	float cubeSize;
	Uint3 cubeCount;
	uint cube_num;
	float isovalue;
#endif

	uint BLOCK;
	uint THREAD;

public:
	Param();
	~Param();

};

