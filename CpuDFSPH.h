// This file is mostly original

#pragma once
#include "Configuration.h"
#ifdef CPU_DF
#include "Type.cuh"
#include "Param.h"
#include <vector>


void Cpu_DFSPHLoop(Particle* dParticles, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd, cube* Cubes, Float3* Triangles, Param* Param,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd);
 void Cpu_DFSPHComputeNormals(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex);
 void Cpu_generateHashTable(Particle* particles, uint* dParticleIndex, uint* dCellIndex, Param* param);
 void Cpu_DFSPHDivergenceSolver2(Param* param, int *isGood);
 void Cpu_DFSPHDivergenceSolver3(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood);
 void Cpu_DFSPHDivergenceSolver1(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood);
 void Cpu_DFSPHCommputeDensityAndFactorAlpha(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd);
void Cpu_find_start_end_kernel(uint *dStart, uint *dEnd, uint *dCellIndex, uint *dParticleIndex, uint num_particle);
 void Cpu_DFSPHDensitySolverPart2(Param* param, int *isGood);
 void Cpu_DFSPHDensitySolverPart3(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood);
 void Cpu_DFSPHPredictDensAndVelocity(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood);
 void Cpu_DFSPHPredictVelocity(Particle* particles, Param* param);
 void Cpu_DFSPHUpdateTimeStep(Particle* particles, Param* param);
 void Cpu_DFSPHComputeVelocityScalar(Particle* particles, Param* param);
 void Cpu_DFSPHComputeSurfaceTensionForce(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd);
 void Cpu_DFSPHComputeForces(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex);
 void Cpu_sort_particles(uint *dHash, uint *dIndex, int num_particle);
void Cpu_DFSPHSetUp(Particle* dParticles, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd, cube* dCubes, Float3* dTriangles, Param* hParam,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd);
 void Cpu_DFSPHUpdatePosition(Particle* particles, Param* param);
 uint Cpu_computeCellHash(Uint3 cellPos, Param* param);
 Uint3 Cpu_computeCellPosition(Float3 pos, Param* param);
 void Cpu_computeBorderPsi(Particle* dParticle, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd);
 void generateHashTable_Boundary(Particle* particles, uint* dParticleIndex, uint* dCellIndex, Param* param);
 void Cpu_MC_Run(cube* dCubes, Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex);
 void Cpu_MC_ComputeNormal(cube* dCubes, Param* param);
 void Cpu_MC_Run2(cube* dCubes, Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex, Float3* dTriangles, Float3* dNorms);
 void Cpu_MC_RUN_ONE_TIME(cube *dCubes, Particle *dParticles, Param *param, uint* dStart, uint* dEnd, uint* dParticleIndex, Float3*dTriangles, Float3* dNorms, Param* hParam);



#endif