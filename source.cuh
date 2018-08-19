// This file is mostly original with some references from the following github repository.
// https://github.com/finallyjustice/sphfluid

#ifndef _KERNEL_CUH
#define _KERNEL_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include "Param.h"
#include "Type.cuh"
#include "PhysicalEngineTimer.h"


void GPU(Particle* dParticles, Param* param, uint* dParticleIndex, uint* dCellIndex,  uint* dStart, uint* dEnd,  cube* dCubes, Float3* dTriangles, Param* hParam);
void DFSPHSetUp(Particle* dParticles, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd, cube* dCubes, Float3* dTriangles, Param* hParam,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd);
void DFSPHLoop(Particle* dParticles, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd, cube* dCubes, Float3* dTriangles, Param* hParam,
			   Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd);
bool initCUDA();
void ComputeBoundaryParticlePsi(Particle* dParticle, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd, Param* hParam);


#ifdef RENDER_MESH
void MC_RUN_ONE_TIME(cube *dCubes, Particle *dParticles, Param *param, uint* dStart, uint* dEnd, uint* dParticleIndex, Float3*dTriangles, Float3* dNorms, Param* hParam);
#endif

#endif

