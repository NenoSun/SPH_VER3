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


// Improvment Code

//class Kernel{
//	public:
//	float support_radius = 0.1f;
//	float coff = 8.0f / PI / pow(support_radius, 3);
//	float grad_coff = (48.0f / PI / pow(support_radius, 4));
//
//	__device__
//	float W(float q){
//		return coff * (6 * pow(q, 3) - 6 * pow(q, 2) + 1);
//	}
//
//	__device__
//	float W(Float3 r){
//		float q = r.Norm() / support_radius;
//		if(q < )
//			return coff * (6 * pow(q, 3) - 6 * pow(q, 2) + 1);
//	}
//
//	__device__
//	Float3 GradW(Float3 r){
//		return grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR / distance;
//	}
//
//};

// Improvment Code

#ifdef RENDER_MESH
void MC_RUN_ONE_TIME(cube *dCubes, Particle *dParticles, Param *param, uint* dStart, uint* dEnd, uint* dParticleIndex, Float3*dTriangles, Float3* dNorms, Param* hParam);
#endif

#endif

