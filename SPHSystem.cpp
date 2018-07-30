#include "SPHSystem.h"

void SPHSystem::PassParamsTo_hPram(Param *param) {
	param->mass = this->parameters.mass;
	param->worldSize = this->parameters.worldSize;
	param->gridSize = this->parameters.gridSize;
	param->cells_total = this->parameters.cells_total;
	param->num_particles = this->parameters.num_particles;
	param->num_boundary_particles = this->parameters.num_particles;
	param->max_num_particles = this->parameters.max_num_particles;
	param->h = this->parameters.h;
	param->h_square = this->parameters.h_square;
	param->l_threshold = this->parameters.l_threshold;

	param->vicosity_coff = this->parameters.vicosity_coff;
	param->gas_stiffness = this->parameters.gas_stiffness;
	param->grad_poly6 = this->parameters.grad_poly6;
	param->grad_spiky_coff = this->parameters.grad_spiky_coff;
	param->lplc_poly6 = this->parameters.lplc_poly6;
	param->lplc_visco_coff = this->parameters.lplc_visco_coff;
	param->poly6_coff = this->parameters.poly6_coff;
	param->spline_coff = this->parameters.spline_coff;
	param->grad_spline_coff = this->parameters.grad_spline_coff;
	param->cohesion_coff = this->parameters.cohesion_coff;
	param->cohesion_term = this->parameters.cohesion_term;
	param->Adhesion_coff = this->parameters.Adhesion_coff;

	param->rest_density = this->parameters.rest_density;
	param->surf_tens_coff = this->parameters.surf_tens_coff;
	param->wall_damping = this->parameters.wall_damping;
	param->timeStep = this->parameters.timeStep;
	param->self_dens = parameters.mass*parameters.poly6_coff*pow(parameters.h, 6);

	param->THREAD = this->parameters.THREAD;
	param->BLOCK = this->parameters.BLOCK;

#ifdef SPLINE_KERNEL
	param->SpeedOfSound = sqrt(2 * 9.8 * WORLDSIZE_Y * UPY) / 0.1;
	param->B = pow(param->SpeedOfSound, 2) / 7.0f;
#endif
	param->radius = this->hParticles[0].radius;

#ifdef RENDER_MESH
	// Marching Cube
	param->cubeCount = this->parameters.cubeCount;
	param->cubeSize = this->parameters.cubeSize;
	param->cubeCount = this->parameters.cubeCount;
	param->cube_num = this->parameters.cube_num;
	param->isovalue = this->parameters.isovalue;
#endif
}

float calDistance(Float3 p1, Float3 p2) {
	return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z);
}

void SPHSystem::computeThreadsAndBlocks(int maxThreadsPerBlock, int num_particles) {
#ifdef LINUX
	int THREADS = min(maxThreadsPerBlock, num_particles);
	int BLOCKS = ceil(num_particles / (float)maxThreadsPerBlock);
	this->THREAD = min(maxThreadsPerBlock, num_particles);
	this->BLOCK = ceil(num_particles / (float)(THREADS));
#endif

#ifdef WINDOWS
	int THREADS = std::fmin(maxThreadsPerBlock, num_particles);
	int BLOCKS = ceil(num_particles / (float)maxThreadsPerBlock);
	this->parameters.THREAD = std::fmin(maxThreadsPerBlock, num_particles);
	this->parameters.BLOCK = ceil(num_particles / (float)(THREADS));
#endif
}


SPHSystem::SPHSystem()
{
	this->parameters.max_num_particles = 500000;
	this->parameters.num_particles = 0;
	this->parameters.num_boundary_particles = 0;
	parameters.worldSize.x = WORLDSIZE_X;
	parameters.worldSize.y = WORLDSIZE_Y;
	parameters.worldSize.z = WORLDSIZE_Z;

	this->sys_running = 0;

#ifdef DF  // This is approach 2 implementation
	this->parameters.mass = 0.1f; // Adapted
	this->parameters.h = 0.1f; // Adapted
	this->parameters.h_square = parameters.h * parameters.h; // Adapted
	this->parameters.rest_density = 1000;  // Adapted
	this->parameters.l_threshold = 7.065f;
	this->parameters.timeStep = 0.000452f;

	this->parameters.gas_stiffness = 50000.0f; // Adapted
	this->parameters.vicosity_coff = 0.03; // Adapted
	this->parameters.surf_tens_coff = 0.2; // Adapted
	this->parameters.wall_damping = -0.5f;

	this->parameters.poly6_coff = 315.0f / (64.0f * PI * pow(parameters.h, 9));
	this->parameters.grad_spiky_coff = -45.0f / (PI * pow(parameters.h, 6));
	this->parameters.lplc_visco_coff = 45.0f / (PI * pow(parameters.h, 6));
	this->parameters.grad_poly6 = -945 / (32 * PI * pow(parameters.h, 9));
	this->parameters.lplc_poly6 = -945 / (32 * PI * pow(parameters.h, 9));

	this->parameters.spline_coff = 8.0f / PI / pow(parameters.h, 3);	// Adapted
	this->parameters.grad_spline_coff = (48.0f / PI / pow(parameters.h, 4));	// Adapted
	this->parameters.cohesion_coff = 32.0f / (PI * pow(parameters.h, 9));
	this->parameters.cohesion_term = pow(parameters.h, 6) / 64.0f;
	this->parameters.Adhesion_coff = 0.007f / pow(parameters.h, 3.25);
#endif

#if defined(KERNEL) || defined(SPLINE_KERNEL) // This is approach 1 implementation
	this->parameters.mass = 0.02f; // Adapted
	this->parameters.h = 0.04f; // Adapted
	this->parameters.h_square = parameters.h*parameters.h; // Adapted
	this->parameters.rest_density = 998;  // Adapted
	this->parameters.l_threshold = 7.065f;
	this->parameters.timeStep = 0.000452f;

	this->parameters.gas_stiffness = 3.0; // Adapted
	this->parameters.vicosity_coff = 3.5; // Adapted
	this->parameters.surf_tens_coff = 0.0728f;
	this->parameters.wall_damping = -0.5f;

	this->parameters.poly6_coff = 315.0f / (64.0f * PI * pow(parameters.h, 9));
	this->parameters.grad_spiky_coff = -45.0f / (PI * pow(parameters.h, 6));
	this->parameters.lplc_visco_coff = 45.0f / (PI * pow(parameters.h, 6));
	this->parameters.grad_poly6 = -945 / (32 * PI * pow(parameters.h, 9));
	this->parameters.lplc_poly6 = -945 / (32 * PI * pow(parameters.h, 9));

	this->parameters.spline_coff = 1.0f / PI / pow(parameters.h, 3);
	this->parameters.grad_spline_coff = 9.0f / (4 * PI * pow(parameters.h, 5));
#endif


	this->isDFSPHReady = false;
	this->parameters.BLOCK = 0;
	this->parameters.THREAD = 0;

	parameters.gridSize.x = (uint)(parameters.worldSize.x / parameters.h);
	parameters.gridSize.y = (uint)(parameters.worldSize.y / parameters.h);
	parameters.gridSize.z = (uint)(parameters.worldSize.z / parameters.h);
	parameters.cells_total = parameters.gridSize.x * parameters.gridSize.y * parameters.gridSize.z;
	//Gravity.x = Gravity.z = 0.0f;
	//Gravity.y = -9.8f;


	hParticles = (Particle*)malloc(sizeof(Particle)*this->parameters.max_num_particles);
	hBoundaryParticles = (Particle*)malloc(sizeof(Particle)*this->parameters.max_num_particles);
#ifdef CPU_DF
	hParticleIndex = (uint*)malloc(sizeof(uint)*this->parameters.max_num_particles);
	hCellIndex = (uint*)malloc(sizeof(uint)*this->parameters.max_num_particles);
	hStart = (uint*)malloc(sizeof(uint)*this->parameters.cells_total);
	hEnd = (uint*)malloc(sizeof(uint)*this->parameters.cells_total);
	hBoundaryStart = (uint*)malloc(sizeof(uint)*this->parameters.cells_total);
	hBoundaryEnd = (uint*)malloc(sizeof(uint)*this->parameters.cells_total);
	hBoundaryParticleIndex = (uint*)malloc(sizeof(uint)*this->parameters.max_num_particles);
	hBoundaryCellIndex = (uint*)malloc(sizeof(uint)*this->parameters.max_num_particles);
#endif


}

SPHSystem::~SPHSystem() {
	cudaFree(dParticles);
	cudaFree(dParam);
	cudaFree(dParticleIndex);
	cudaFree(dCellIndex);
	cudaFree(dStart);
	cudaFree(dEnd);
#ifdef RENDER_MESH
	cudaFree(dCubes);
	cudaFree(dTriangles);
#endif
}


#ifdef RENDER_MESH
void SPHSystem::MarchingCubeSetUp() {
	parameters.cubePerAxies = MESH_RESOLUTION; // The minimal cube number on an axies
#ifdef LINUX
	cubeSize = min(WORLDSIZE_X, min(WORLDSIZE_Y, WORLDSIZE_Z)) / float(cubePerAxies); // The length of the cube
#endif
#ifdef WINDOWS
	parameters.cubeSize = std::fmin(WORLDSIZE_X, std::fmin(WORLDSIZE_Y, WORLDSIZE_Z)) / float(parameters.cubePerAxies); // The length of the cube
#endif
	parameters.cubeCount.x = WORLDSIZE_X / parameters.cubeSize; // Cube number on X-axies
	parameters.cubeCount.y = WORLDSIZE_Y / parameters.cubeSize; // Cube number on Y-axies
	parameters.cubeCount.z = WORLDSIZE_Z / parameters.cubeSize; // Cube number on Z-axies

	if (ceil(parameters.cubeCount.x) - parameters.cubeCount.x > 1e-4)
		parameters.cubeCount.x = floor(parameters.cubeCount.x);
	else
		parameters.cubeCount.x = ceil(parameters.cubeCount.x);

	if (ceil(parameters.cubeCount.y) - parameters.cubeCount.y > 1e-4)
		parameters.cubeCount.y = floor(parameters.cubeCount.y);
	else
		parameters.cubeCount.y = ceil(parameters.cubeCount.y);

	if (ceil(parameters.cubeCount.z) - parameters.cubeCount.z > 1e-4)
		parameters.cubeCount.z = floor(parameters.cubeCount.z);
	else
		parameters.cubeCount.z = ceil(parameters.cubeCount.z);

	parameters.cube_num = parameters.cubeCount.x * parameters.cubeCount.y * parameters.cubeCount.z;
	this->parameters.isovalue = IOSVALUE;


	printf("CubeCount: %d, %d, %d\n", parameters.cubeCount.x, parameters.cubeCount.y, parameters.cubeCount.z);
	printf("CubeSize: %f\n", parameters.cubeSize);
	printf("Cube Num: %d\n", parameters.cube_num);


	hCubes = (cube*)malloc(sizeof(cube)*parameters.cube_num);
	hTriangles = (Float3*)malloc(sizeof(Float3) * 15 * parameters.cube_num);
	hNorms = (Float3*)malloc(sizeof(Float3) * 15 * parameters.cube_num);

	cudaMalloc((void**)&dCubes, sizeof(cube)*parameters.cube_num);
	cudaMalloc((void**)&dTriangles, sizeof(Float3) * 15 * parameters.cube_num);
	cudaMalloc((void**)&dNorms, sizeof(Float3) * 15 * parameters.cube_num);

	uint offset[8][3] = {
		{ 0, 0, 0 },{ 1, 0, 0 },{ 1, 1, 0 },{ 0, 1, 0 },
		{ 0, 0, 1 },{ 1, 0, 1 },{ 1, 1, 1 },{ 0, 1, 1 }
	};

	// Initalize the vertices in each cube
	for (int i = 0; i < parameters.cubeCount.x; i++)
		for (int j = 0; j < parameters.cubeCount.y; j++)
			for (int k = 0; k < parameters.cubeCount.z; k++) {
				cube* c = &hCubes[CUBE_HASH(i, j, k)];
				for (int w = 0; w < 8; w++) {
					c->vertices[w].pos.x = (i + offset[w][0])*parameters.cubeSize;
					c->vertices[w].pos.y = (j + offset[w][1])*parameters.cubeSize;
					c->vertices[w].pos.z = (k + offset[w][2])*parameters.cubeSize;
				}
			}

	cudaMemcpy(dCubes, hCubes, sizeof(cube)*this->parameters.cube_num, cudaMemcpyHostToDevice);
}

#endif

// CPU Function
void SPHSystem::generateParticles() {
	Float3 pos;
	Float3 vel;
	vel.x = vel.y = vel.z = 0.0f;

	for (pos.x = parameters.worldSize.x*DOWNX; pos.x < parameters.worldSize.x*UPX; pos.x = pos.x + parameters.h*INTERVAL)
		for (pos.y = parameters.worldSize.y*DOWNY; pos.y < parameters.worldSize.y*UPY; pos.y = pos.y + parameters.h*INTERVAL)
			for (pos.z = parameters.worldSize.z*DOWNZ; pos.z < parameters.worldSize.z*UPZ; pos.z = pos.z + parameters.h*INTERVAL) {
				addParticle(pos, vel);
			}

#ifdef ENABLE_BOUNDARY_PARTICLE
	// The boundary particle position is a small positive number more than zero or a number that is slightly smaller than the boundary to avoid Rounding Error
	pos.y = BOUNDARY;
	for (pos.x = 0.0f; pos.x < parameters.worldSize.x; pos.x = pos.x + parameters.h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.z = 0.0f; pos.z < parameters.worldSize.z; pos.z = pos.z + parameters.h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.y = parameters.worldSize.y - BOUNDARY;
	for (pos.x = 0.0f; pos.x < parameters.worldSize.x; pos.x = pos.x + parameters.h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.z = 0.0f; pos.z < parameters.worldSize.z; pos.z = pos.z + parameters.h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.x = BOUNDARY;
	for (pos.y = 0.0f; pos.y < parameters.worldSize.y; pos.y = pos.y + parameters.h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.z = 0.0f; pos.z < parameters.worldSize.z; pos.z = pos.z + parameters.h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.x = parameters.worldSize.x - BOUNDARY;
	for (pos.y = 0.0f; pos.y < parameters.worldSize.y; pos.y = pos.y + parameters.h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.z = 0.0f; pos.z < parameters.worldSize.z; pos.z = pos.z + parameters.h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.z = BOUNDARY;
	for (pos.y = 0.0f; pos.y < parameters.worldSize.y; pos.y = pos.y + parameters.h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.x = 0.0f; pos.x < parameters.worldSize.x; pos.x = pos.x + parameters.h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.z = parameters.worldSize.z - BOUNDARY;
	for (pos.y = 0.0f; pos.y < parameters.worldSize.y; pos.y = pos.y + parameters.h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.x = 0.0f; pos.x < parameters.worldSize.x; pos.x = pos.x + parameters.h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);
#endif

	// Spherical Object
	//Float3 center;
	//center.x = 0.5 * worldSize.x;
	//center.y = 0.0 * worldSize.y;
	//center.z = 0.25 * worldSize.z;
	//addObject("/home/neno/SPlisHSPlasH/data/models/Dragon_50k.obj", center);

	//center.z = 0.75 * worldSize.z;
	//addObject("/home/neno/SPlisHSPlasH/data/models/Dragon_50k.obj", center);

	//center.z = 0.5 * worldSize.z;
	//center.x = 0.85 * worldSize.x;
	//addObject("/home/neno/SPlisHSPlasH/data/models/Dragon_50k.obj", center);

	//addSphericalObject(center, 0.1 * worldSize.x);

	//center.x = 0.2 * worldSize.x;
	//center.y = 0.2 * worldSize.y;
	//center.z = 0.2 * worldSize.z;

	//addSphericalObject(center, 0.1 * worldSize.x);

	//center.x = 0.7 * worldSize.x;
	//center.y = 0.2 * worldSize.y;
	//center.z = 0.7 * worldSize.z;
	//
	//addSphericalObject(center, 0.1 * worldSize.x);

	printf("Partcile num: %d\n", parameters.num_particles);
	printf("Boundary Particle Num: %d\n", parameters.num_boundary_particles);

	//GPU Initalizaiton
	initCUDA();

	// Allocate host memory
	hParam = (Param*)malloc(sizeof(Param));

	// Allocate GPU memory
	cudaMalloc((void**)(&dParticles), sizeof(Particle)*this->parameters.num_particles);
	cudaMalloc((void**)(&dBoundaryParticles), sizeof(Particle)*this->parameters.num_boundary_particles);
	cudaMalloc((void**)(&dParam), sizeof(Param));
	cudaMalloc((void**)&dParticleIndex, sizeof(uint)*this->parameters.num_particles);
	cudaMalloc((void**)&dCellIndex, sizeof(uint)*this->parameters.num_particles);
	cudaMalloc((void**)&dBoundaryParticleIndex, sizeof(uint)*this->parameters.num_boundary_particles);
	cudaMalloc((void**)&dBoundaryCellIndex, sizeof(uint)*this->parameters.num_boundary_particles);
	cudaMalloc((void**)&dStart, sizeof(uint)*this->parameters.cells_total);
	cudaMalloc((void**)&dEnd, sizeof(uint)*this->parameters.cells_total);
	cudaMalloc((void**)&dBoundaryStart, sizeof(uint)*this->parameters.cells_total);
	cudaMalloc((void**)&dBoundaryEnd, sizeof(uint)*this->parameters.cells_total);

#ifdef RENDER_MESH
	this->MarchingCubeSetUp();
#endif

	computeThreadsAndBlocks(600, this->parameters.num_particles);

	PassParamsTo_hPram(hParam);

	// Copy initial data
	cudaMemcpy(dParam, hParam, sizeof(Param), cudaMemcpyHostToDevice);
	cudaMemcpy(dParticles, this->hParticles, sizeof(Particle)*parameters.num_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(dBoundaryParticles, this->hBoundaryParticles, sizeof(Particle)*parameters.num_boundary_particles, cudaMemcpyHostToDevice);

#ifndef CPU_DF
	ComputeBoundaryParticlePsi(dBoundaryParticles, dParam, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, hParam);
#else
	for (int i = 0; i < hParam->num_boundary_particles; i++) {
		hBoundaryCellIndex[i] = 0xffffffff;
		hBoundaryParticleIndex[i] = 0xffffffff;
	}
	for (int i = 0; i < parameters.cells_total; i++) {
		hBoundaryStart[i] = 0xffffffff;
		hBoundaryEnd[i] = 0xffffffff;
	}
	generateHashTable_Boundary(hBoundaryParticles, hBoundaryParticleIndex, hBoundaryCellIndex, hParam);
	Cpu_sort_particles(hBoundaryCellIndex, hBoundaryParticleIndex, hParam->num_boundary_particles);
	Cpu_find_start_end_kernel(hBoundaryStart, hBoundaryEnd, hBoundaryCellIndex, hBoundaryParticleIndex, hParam->num_boundary_particles);

	Cpu_computeBorderPsi(hBoundaryParticles, hParam, hBoundaryParticleIndex, hBoundaryCellIndex, hBoundaryStart, hBoundaryEnd);
#endif


	printf("BLOCK:%d, THREADS:%d\n", parameters.BLOCK, parameters.THREAD);
}

void SPHSystem::addParticle(Float3 pos, Float3 vel) {
	// Create a new particle
	Particle *p = &(hParticles[parameters.num_particles]);

	// Assign it identity
	parameters.num_particles++;

	// Assgin it with spatial and physical properties
	p->pos = pos;
	p->vel = vel;
	p->acc.x = p->acc.y = p->acc.z = 0;
	p->dens = parameters.rest_density;
	p->predict_dens = parameters.rest_density;
	p->pres = 0.0f;
	p->alpha = 0.0f;
	p->grad_dens = 0.0f;
	p->radius = 0.025f; // Changed
	p->Psi = 0;
	p->norm.x = p->norm.y = p->norm.z = 0.0f;
	p->isObject = false;
}

void SPHSystem::addBoundaryParticle(Float3 pos, Float3 vel, bool isObject) {
	// Find the next empty particle slot
	Particle *p = &(hBoundaryParticles[parameters.num_boundary_particles]);

	// The pointer increments and points to the next empty particle slot
	parameters.num_boundary_particles++;

	// Assgin it with spatial and physical properties
	p->pos = pos;
	p->vel = vel;
	p->acc.x = p->acc.y = p->acc.z = 0;
	p->dens = parameters.rest_density;
	p->pres = 0.0f;
	p->alpha = 0.0f;
	p->grad_dens = 0.0f;
	p->radius = 0.025f; // Changed
	p->Psi = 0;
	p->norm.x = p->norm.y = p->norm.z = 0.0f;
	p->isObject = isObject;
}

#ifdef MARCHING_CUBES_TIMER
PhysicalEngineTimer MCTimer(MARCHING_CUBES_TIMEDATA_FILEPATH);
#endif

void SPHSystem::animation() {
	if (sys_running == 0)
		return;

	// Divergence-Free SPH Simulation
#ifdef DF
#ifdef CPU_DF
	if (!isDFSPHReady) {
		Cpu_DFSPHSetUp(this->hParticles, this->hParam, hParticleIndex, hCellIndex, hStart, hEnd, hCubes, hTriangles, hParam,
			hBoundaryParticles, hBoundaryParticleIndex, hBoundaryCellIndex, hBoundaryStart, hBoundaryEnd);
	}
	Cpu_DFSPHLoop(this->hParticles, this->hParam, hParticleIndex, hCellIndex, hStart, hEnd, hCubes, hTriangles, hParam,
		hBoundaryParticles, hBoundaryParticleIndex, hBoundaryCellIndex, hBoundaryStart, hBoundaryEnd);
#else
	if (!isDFSPHReady) {
		DFSPHSetUp(dParticles, dParam, dParticleIndex, dCellIndex, dStart, dEnd, dCubes, dTriangles, hParam,
			dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd);
		this->isDFSPHReady = true;
	}
	DFSPHLoop(dParticles, dParam, dParticleIndex, dCellIndex, dStart, dEnd, dCubes, dTriangles, hParam,
		dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd);
#endif


	// Normal SPH simulation
#else
	GPU(dParticles, dParam, dParticleIndex, dCellIndex, dStart, dEnd, dCubes, dTriangles, hParam);
#endif

	// Copy the data from GPU to host
#ifndef CPU_DF
	cudaMemcpy(hParticles, dParticles, sizeof(Particle)*hParam->num_particles, cudaMemcpyDeviceToHost);
#endif


#ifdef RENDER_MESH
#ifdef MARCHING_CUBES_TIMER
	MCTimer.start();
#endif
#ifdef CPU_DF
	Cpu_MC_RUN_ONE_TIME(hCubes, hParticles, hParam, hStart, hEnd, hParticleIndex, hTriangles, hNorms, hParam);
#else
	MC_RUN_ONE_TIME(dCubes, dParticles, dParam, dStart, dEnd, dParticleIndex, dTriangles, dNorms, hParam);
#ifdef MARCHING_CUBES_TIMER
	MCTimer.end();
#endif
	cudaMemcpy(hTriangles, dTriangles, sizeof(Float3)*parameters.cube_num * 15, cudaMemcpyDeviceToHost);
	cudaMemcpy(hNorms, dNorms, sizeof(Float3)*parameters.cube_num * 15, cudaMemcpyDeviceToHost);
#endif
#endif
}


void SPHSystem::addSphericalObject(Float3 center, float radius) {
	// Spherical Object
	Float3 pos;
	Float3 vel;
	vel.x = 0; vel.y = 0; vel.z = 0;

	for (pos.x = -radius; pos.x < radius; pos.x = pos.x + parameters.h * BOUNDARY_PARTICLE_INTERVAL)
		for (pos.z = -radius; pos.z < radius; pos.z = pos.z + parameters.h * BOUNDARY_PARTICLE_INTERVAL) {
			float norm = radius * radius - pos.x * pos.x - pos.z * pos.z;
			if (norm < 0) {
				printf("Bad particle : %f.\n", norm);
				continue;
			}
			pos.y = sqrtf(norm);
			printf("Two particles are adding into the system...\n");
			addBoundaryParticle(center + pos, vel, true);
			addBoundaryParticle(center - pos, vel, true);
			printf("Add complete.\n");
		}
}

void SPHSystem::addObject(string url, Float3 pos) {
	// reading a text file

	using namespace std;
	string line;
	ifstream myfile(url);
	Float3 temp;
	Float3 vel;
	temp.x = -99.0f; temp.y = -99.0f; temp.z = -99.0f;
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			if (line.substr(0, 1) == "v") {
				istringstream iss(line);
				do {
					string subs;
					iss >> subs;
					if (subs == "" || subs == "v")
						continue;
					if (temp.x == -99.0&&temp.y == -99.0&&temp.z == -99.0) {
						temp.x = stof(subs);
						continue;
					}
					else if (temp.x != -99.0&&temp.y == -99.0&&temp.z == -99.0) {
						temp.y = stof(subs);
						continue;
					}
					else if (temp.x != -99.0&&temp.y != -99.0&&temp.z == -99.0) {
						temp.z = stof(subs);
						continue;
					}
				} while (iss);
				addBoundaryParticle(temp + pos, vel, true);
				temp.x = -99.0f; temp.y = -99.0f; temp.z = -99.0f;
			}
		}
		myfile.close();
	}
	else cout << "Unable to open file";
}
