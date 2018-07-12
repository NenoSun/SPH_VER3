#include "SPHSystem.h"


//#define WORLDSIZE 10.0f
//#define WORLDSIZE 4.0f
//#define WORLDSIZE .64f

// DAM BREAK CONFIGURATION
//#define DOWNX .01f  
//#define UPX .30f  
//#define DOWNY .01f
//#define UPY .40f
//#define DOWNZ .01f
//#define UPZ 0.99f

// WATER DROP CONFIGURATION


//#define INTERVAL .35f

void SPHSystem::PassParamsTo_hPram(Param *param) {
	param->mass = this->mass;
	param->worldSize = this->worldSize;
	param->gridSize = this->gridSize;
	param->Gravity = this->Gravity;
	param->cells_total = this->grid_num;
	param->num_particles = this->num_particles;
	param->num_boundary_particles = this->num_boundary_p;
	param->max_num_particles = this->num_max;
	param->h = this->h;
	param->h_square = this->h_square;
	param->l_threshold = this->l_threshold;

	param->vicosity_coff = this->vicosity_coff;
	param->gas_stiffness = this->gas_stiffness;
	param->grad_poly6 = this->grad_poly6;
	param->grad_spiky_coff = this->grad_spiky_coff;
	param->lplc_poly6 = this->lplc_poly6;
	param->lplc_visco_coff = this->lplc_visco_coff;
	param->poly6_coff = this->poly6_coff;
	param->spline_coff = this->spline_coff;
	param->grad_spline_coff = this->grad_spline_coff;
	param->cohesion_coff = this->cohesion_coff;
	param->cohesion_term = this->cohesion_term;
	param->Adhesion_coff = this->Adhesion_coff;

	param->rest_density = this->rest_density;
	param->surf_tens_coff = this->surf_tens_coff;
	param->wall_damping = this->wall_damping;
	param->timeStep = this->timeStep;
	param->self_dens = mass*poly6_coff*pow(h, 6);

	param->THREAD = this->THREAD;
	param->BLOCK = this->BLOCK;

	//param->SpeedOfSound = sqrt(2 * 9.8 * WORLDSIZE * UPY) / 0.1;
	//param->B = pow(param->SpeedOfSound, 2) / 7.0f;
	param->radius = this->hParticles[0].radius;

	param->flag = false;

#ifdef RENDER_MESH
	// Marching Cube
	param->cubeCount = this->cubeCount;
	param->cubeSize = this->cubeSize;
	param->cubeCount = this->cubeCount;
	param->cube_num = this->cube_num;
	param->isovalue = this->isovalue;
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
	this->THREAD = std::fmin(maxThreadsPerBlock, num_particles);
	this->BLOCK = ceil(num_particles / (float)(THREADS));
#endif
}


SPHSystem::SPHSystem()
{
	this->num_max = 500000; 
	this->num_particles = 0;
	this->num_boundary_p = 0;
	worldSize.x = WORLDSIZE_X;
	worldSize.y = WORLDSIZE_Y;
	worldSize.z = WORLDSIZE_Z;

	this->sys_running = 0;

#ifdef DF  // This is approach 2 implementation
	this->mass = 0.1f; // Adapted
	this->h = 0.1f; // Adapted
	this->h_square = h*h; // Adapted
	this->rest_density = 1000;  // Adapted
	this->l_threshold = 7.065f;
	this->timeStep = 0.000452f;

	this->gas_stiffness = 50000.0f; // Adapted
	this->vicosity_coff = 0.03; // Adapted
	this->surf_tens_coff = 0.2; // Adapted
	this->wall_damping = -0.5f;

	this->poly6_coff = 315.0f / (64.0f * PI * pow(h, 9));
	this->grad_spiky_coff = -45.0f / (PI * pow(h, 6));
	this->lplc_visco_coff = 45.0f / (PI * pow(h, 6));
	this->grad_poly6 = -945 / (32 * PI * pow(h, 9));
	this->lplc_poly6 = -945 / (32 * PI * pow(h, 9));

	this->spline_coff = 8.0f / PI / pow(h, 3);	// Adapted
	this->grad_spline_coff = (48.0f / PI / pow(h, 4));	// Adapted
	this->cohesion_coff = 32.0f / (PI * pow(h, 9));
	this->cohesion_term = pow(h, 6) / 64.0f;
	this->Adhesion_coff = 0.007f/ pow(h, 3.25);
#endif

#ifdef KERNEL // This is approach 1 implementation
	this->mass = 0.02f; // Adapted
	this->h = 0.04f; // Adapted
	this->h_square = h*h; // Adapted
	this->rest_density = 998;  // Adapted
	this->l_threshold = 7.065f;
	this->timeStep = 0.000452f;

	this->gas_stiffness = 3.0; // Adapted
	this->vicosity_coff = 3.5; // Adapted
	this->surf_tens_coff = 0.0728f;
	this->wall_damping = -0.5f;

	this->poly6_coff = 315.0f / (64.0f * PI * pow(h, 9));
	this->grad_spiky_coff = -45.0f / (PI * pow(h, 6));
	this->lplc_visco_coff = 45.0f / (PI * pow(h, 6));
	this->grad_poly6 = -945 / (32 * PI * pow(h, 9));
	this->lplc_poly6 = -945 / (32 * PI * pow(h, 9));

	this->spline_coff = 1.0f / PI / pow(h, 3);
	this->grad_spline_coff = 9.0f / (4 * PI * pow(h, 5));
#endif


	this->isDFSPHReady = false;
	this->BLOCK = 0;
	this->THREAD = 0;

	

	hParticles = (Particle*)malloc(sizeof(Particle)*this->num_max);
	hBoundaryParticles = (Particle*)malloc(sizeof(Particle)*this->num_max);
	gridSize.x = (uint)(worldSize.x / h);
	gridSize.y = (uint)(worldSize.y / h);
	gridSize.z = (uint)(worldSize.z / h);
	grid_num = gridSize.x * gridSize.y * gridSize.z;
	Gravity.x = Gravity.z = 0.0f;
	Gravity.y = -9.8f;
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



void SPHSystem::MarchingCubeSetUp() {
	cubePerAxies = MESH_RESOLUTION; // The minimal cube number on an axies
	cubeSize= min(WORLDSIZE_X, min(WORLDSIZE_Y, WORLDSIZE_Z)) / float(cubePerAxies); // The length of the cube
	cubeCount.x = WORLDSIZE_X / cubeSize; // Cube number on X-axies
	cubeCount.y = WORLDSIZE_Y / cubeSize; // Cube number on Y-axies
	cubeCount.z = WORLDSIZE_Z / cubeSize; // Cube number on Z-axies

	if(ceil(cubeCount.x) - cubeCount.x > 1e-4)
		cubeCount.x = floor(cubeCount.x);
	else
		cubeCount.x = ceil(cubeCount.x);

	if(ceil(cubeCount.y) - cubeCount.y > 1e-4)
		cubeCount.y = floor(cubeCount.y);
	else
		cubeCount.y = ceil(cubeCount.y);

	if(ceil(cubeCount.z) - cubeCount.z > 1e-4)
		cubeCount.z = floor(cubeCount.z);
	else
		cubeCount.z = ceil(cubeCount.z);

	cube_num = cubeCount.x * cubeCount.y * cubeCount.z;
	this->isovalue = IOSVALUE;


	printf("CubeCount: %d, %d, %d\n", cubeCount.x, cubeCount.y, cubeCount.z);
	printf("CubeSize: %f\n", cubeSize);
	printf("Cube Num: %d\n", cube_num);

	// Give up CPU Version
	hCubes = (cube*)malloc(sizeof(cube)*cube_num);
	hTriangles = (Float3*)malloc(sizeof(Float3) * 15 * cube_num);
	hNorms = (Float3*)malloc(sizeof(Float3) * 15 * cube_num);

	cudaMalloc((void**)&dCubes, sizeof(cube)*cube_num);
	cudaMalloc((void**)&dTriangles, sizeof(Float3) * 15 * cube_num);
	cudaMalloc((void**)&dNorms, sizeof(Float3) * 15 * cube_num);

	uint offset[8][3] = {
		{ 0, 0, 0 },{ 1, 0, 0 },{ 1, 1, 0 },{ 0, 1, 0 },
		{ 0, 0, 1 },{ 1, 0, 1 },{ 1, 1, 1 },{ 0, 1, 1 }
	};
	
	// Initalize the vertices in each cube
	for (int i = 0; i < cubeCount.x; i++)
		for (int j = 0; j < cubeCount.y; j++)
			for (int k = 0; k < cubeCount.z; k++) {
				cube* c = &hCubes[CUBE_HASH(i, j, k)];
				for (int w = 0; w < 8; w++) {
					c->vertices[w].pos.x = (i + offset[w][0])*cubeSize;
					c->vertices[w].pos.y = (j + offset[w][1])*cubeSize;
					c->vertices[w].pos.z = (k + offset[w][2])*cubeSize;
				}
			}
	
	cudaMemcpy(dCubes, hCubes, sizeof(cube)*this->cube_num, cudaMemcpyHostToDevice);
}	

// CPU Function
void SPHSystem::generateParticles() {
	Float3 pos;
	Float3 vel;
	vel.x = vel.y = vel.z = 0.0f;

	for (pos.x = worldSize.x*DOWNX; pos.x < worldSize.x*UPX; pos.x = pos.x + h*INTERVAL)
		for (pos.y = worldSize.y*DOWNY; pos.y < worldSize.y*UPY; pos.y = pos.y +  h*INTERVAL)
			for (pos.z = worldSize.z*DOWNZ; pos.z < worldSize.z*UPZ; pos.z = pos.z +  h*INTERVAL) {
				addParticle(pos, vel);
			}

	// The boundary particle position is a small positive number more than zero or a number that is slightly smaller than the boundary to avoid Rounding Error
	pos.y = BOUNDARY;
	for (pos.x = 0.0f; pos.x < worldSize.x; pos.x = pos.x + h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.z = 0.0f; pos.z < worldSize.z; pos.z = pos.z + h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.y = worldSize.y - BOUNDARY;
	for (pos.x = 0.0f; pos.x < worldSize.x; pos.x = pos.x + h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.z = 0.0f; pos.z < worldSize.z; pos.z = pos.z + h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.x = BOUNDARY;
	for (pos.y = 0.0f; pos.y < worldSize.y; pos.y = pos.y + h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.z = 0.0f; pos.z < worldSize.z; pos.z = pos.z + h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.x = worldSize.x - BOUNDARY;
	for (pos.y = 0.0f; pos.y < worldSize.y; pos.y = pos.y + h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.z =0.0f; pos.z < worldSize.z; pos.z = pos.z + h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.z = BOUNDARY;
	for (pos.y = 0.0f; pos.y < worldSize.y; pos.y = pos.y + h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.x = 0.0f; pos.x < worldSize.x; pos.x = pos.x + h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

	pos.z = worldSize.z - BOUNDARY;
	for (pos.y = 0.0f; pos.y < worldSize.y; pos.y = pos.y + h* BOUNDARY_PARTICLE_INTERVAL)
		for (pos.x = 0.0f; pos.x < worldSize.x; pos.x = pos.x + h*BOUNDARY_PARTICLE_INTERVAL)
			addBoundaryParticle(pos, vel, false);

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

	printf("Partcile num: %d\n", num_particles);
	printf("Boundary Particle Num: %d\n", num_boundary_p);

	//GPU Initalizaiton
	initCUDA();

	// Allocate host memory
	hParam = (Param*)malloc(sizeof(Param));

	// Allocate GPU memory
	cudaMalloc((void**)(&dParticles), sizeof(Particle)*this->num_particles);
	cudaMalloc((void**)(&dBoundaryParticles), sizeof(Particle)*this->num_boundary_p);
	cudaMalloc((void**)(&dParam), sizeof(Param));
	cudaMalloc((void**)&dParticleIndex, sizeof(uint)*this->num_particles);
	cudaMalloc((void**)&dCellIndex, sizeof(uint)*this->num_particles);
	cudaMalloc((void**)&dBoundaryParticleIndex, sizeof(uint)*this->num_boundary_p);
	cudaMalloc((void**)&dBoundaryCellIndex, sizeof(uint)*this->num_boundary_p);
	cudaMalloc((void**)&dStart, sizeof(uint)*this->grid_num);
	cudaMalloc((void**)&dEnd, sizeof(uint)*this->grid_num);
	cudaMalloc((void**)&dBoundaryStart, sizeof(uint)*this->grid_num);
	cudaMalloc((void**)&dBoundaryEnd, sizeof(uint)*this->grid_num);

#ifdef RENDER_MESH
	this->MarchingCubeSetUp();
#endif

	computeThreadsAndBlocks(600, this->num_particles);

	PassParamsTo_hPram(hParam);

	// Copy initial data
	cudaMemcpy(dParam, hParam, sizeof(Param), cudaMemcpyHostToDevice);
	cudaMemcpy(dParticles, this->hParticles, sizeof(Particle)*num_particles, cudaMemcpyHostToDevice);
	cudaMemcpy(dBoundaryParticles, this->hBoundaryParticles, sizeof(Particle)*num_boundary_p, cudaMemcpyHostToDevice);

	ComputeBoundaryParticlePsi(dBoundaryParticles, dParam, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, hParam);

	printf("BLOCK:%d, THREADS:%d\n", BLOCK, THREAD);
}

void SPHSystem::addParticle(Float3 pos, Float3 vel) {
	// Create a new particle
	Particle *p = &(hParticles[num_particles]);

	// Assign it identity
	num_particles++;

	// Assgin it with spatial and physical properties
	p->pos = pos;
	p->vel = vel;
	p->acc.x = p->acc.y = p->acc.z = 0;
	p->dens = rest_density;
	p->predict_dens = rest_density;
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
	Particle *p = &(hBoundaryParticles[num_boundary_p]);

	// The pointer increments and points to the next empty particle slot
	num_boundary_p++;

	// Assgin it with spatial and physical properties
	p->pos = pos;
	p->vel = vel;
	p->acc.x = p->acc.y = p->acc.z = 0;
	p->dens = rest_density;
	p->pres = 0.0f;
	p->alpha = 0.0f;
	p->grad_dens = 0.0f;
	p->radius = 0.025f; // Changed
	p->Psi = 0;
	p->norm.x = p->norm.y = p->norm.z = 0.0f;
	p->isObject = isObject;
}

void SPHSystem::animation() {
	if (sys_running == 0)
		return;

// Divergence-Free SPH Simulation
#ifdef DF
	if (!isDFSPHReady) {
		DFSPHSetUp(dParticles, dParam, dParticleIndex, dCellIndex, dStart, dEnd, dCubes, dTriangles, hParam,
			dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd);
		this->isDFSPHReady = true;
	}
	DFSPHLoop(dParticles, dParam, dParticleIndex, dCellIndex, dStart, dEnd, dCubes, dTriangles, hParam,
			  dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd);

// Normal SPH simulation
#else
	GPU(dParticles, dParam, dParticleIndex, dCellIndex, dStart, dEnd, dCubes, dTriangles, hParam);
#endif

// Copy the data from GPU to host
	cudaMemcpy(hParticles, dParticles, sizeof(Particle)*hParam->num_particles, cudaMemcpyDeviceToHost);


#ifdef RENDER_MESH
		MC_RUN_ONE_TIME(dCubes, dParticles, dParam, dStart, dEnd, dParticleIndex, dTriangles, dNorms, hParam);
		cudaMemcpy(hTriangles, dTriangles, sizeof(Float3)*cube_num * 15, cudaMemcpyDeviceToHost);
		cudaMemcpy(hNorms, dNorms, sizeof(Float3)*cube_num * 15, cudaMemcpyDeviceToHost);
#endif
}

void SPHSystem::MarchingCubeRun() {
}

void SPHSystem::addSphericalObject(Float3 center, float radius){
	// Spherical Object
	Float3 pos;
	Float3 vel;
	vel.x = 0; vel.y = 0; vel.z = 0;

	for(pos.x = -radius; pos.x < radius; pos.x = pos.x + h * BOUNDARY_PARTICLE_INTERVAL)
		for(pos.z = -radius; pos.z < radius; pos.z = pos.z + h * BOUNDARY_PARTICLE_INTERVAL){
			float norm = radius * radius - pos.x * pos.x - pos.z * pos.z;
			if(norm < 0){
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

void SPHSystem::addObject(string url, Float3 pos){
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
