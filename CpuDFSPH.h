#include "Configuration.h"
#ifndef __CPUSPH__
#define __CPUSPH__
#include "Type.cuh"
#include "Param.h"
#include <vector>

static void Cpu_DFSPHLoop(Particle* dParticles, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd, cube* dCubes, Float3* dTriangles, Param* hParam,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd);
static void Cpu_DFSPHComputeNormals(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex);
static void generateHashTable(Particle* particles, uint* dParticleIndex, uint* dCellIndex, Param* param);

#endif