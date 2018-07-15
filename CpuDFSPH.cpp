#include "CpuDFSPH.h"

Uint3 computeCellPosition(Float3 pos, Param* param) {
	Uint3 cellPos;
	cellPos.x = (uint)floor(pos.x / param->h);
	cellPos.y = (uint)floor(pos.y / param->h);
	cellPos.z = (uint)floor(pos.z / param->h);
	return cellPos;
}

uint computeCellHash(Uint3 cellPos, Param* param) {
	if (cellPos.x>param->gridSize.x - 1 || cellPos.y>param->gridSize.y - 1 || cellPos.z>param->gridSize.z - 1)
		return -1;

	return (uint)(cellPos.z*param->gridSize.x*param->gridSize.y + cellPos.y*param->gridSize.x + cellPos.x);
}

static void Cpu_DFSPHLoop(Particle* dParticles, Param* param, uint* dParticleIndex, uint* dCellIndex, uint* dStart, uint* dEnd, cube* Cubes, Float3* Triangles, Param* Param,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd) {
	Cpu_DFSPHComputeNormals(dParticles, param, dStart, dEnd, dParticleIndex);
	Cpu_DFSPHComputeForces(dParticles, param, dStart, dEnd, dParticleIndex);
	Cpu_DFSPHComputeSurfaceTensionForce(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd);
	Cpu_DFSPHComputeVelocityScalar(dParticles, param);
	Cpu_DFSPHUpdateTimeStep(dParticles, param);
	Cpu_DFSPHPredictVelocity(dParticles, param);
	int counter = 0;
	int* dIsGood = 0;
	while ((dIsGood == 0 || counter < 2) && counter < 100) {
		param->avg_dens = 0.0f;
		Cpu_DFSPHPredictDensAndVelocity(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, dIsGood);
		Cpu_DFSPHDensitySolverPart3(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, dIsGood);
		Cpu_DFSPHDensitySolverPart2(param, dIsGood);
		printf("DENSITY SOLVER ITERATION %d\n", counter);
		counter++;
	}
	for (int i = 0; i < param->num_particles; i++) {
		dCellIndex[i] = 0xffffffff;
		dParticleIndex[i] = 0xffffffff;
	}
	for (int i = 0; i < param->cells_total; i++) {
		dStart[i] = 0xffffffff;
		dEnd[i] = 0xffffffff;
	}
	generateHashTable(dParticles, dParticleIndex, dCellIndex, param);
	Cpu_sort_particles(dCellIndex, dParticleIndex, param->num_particles);
	Cpu_find_start_end_kernel(dStart, dEnd, dCellIndex, dParticleIndex, param->num_particles);
	Cpu_find_start_end_kernel(dStart, dEnd, dCellIndex, dParticleIndex, param->num_particles);
	Cpu_DFSPHCommputeDensityAndFactorAlpha(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd);

	*dIsGood = 0;
	counter = 0;
	while ((dIsGood == 0 || counter < 1) && counter < 100) {
		param->avg_grad_dens = 0.0f;
		Cpu_DFSPHDivergenceSolver1(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, dIsGood);
		Cpu_DFSPHDivergenceSolver3(dParticles, param, dStart, dEnd, dParticleIndex, dBoundaryParticles, dBoundaryParticleIndex, dBoundaryCellIndex, dBoundaryStart, dBoundaryEnd, dIsGood);
		Cpu_DFSPHDivergenceSolver2(param, dIsGood);
		printf("DIVERGENCE SOLVER ITERATION %d\n", counter);
		counter++;
	}
}

static void Cpu_DFSPHComputeNormals(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex) {
	for (int index = 0; index < param->num_particles; index++) {
		Uint3 neighborPos;
		Uint3 cellPos = computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		p->norm.x = p->norm.y = p->norm.z = 0.0f;
		uint count = 0;

		ITERATE_NEIGHBOR{
		neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		int hash = computeCellHash(neighborPos, param);
		if (hash < 0 || hash >= param->cells_total)
			continue;
		// If there exists particles in the cell_index
		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *current = &particles[dParticleIndex[count]];
				float distance = current->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Particle j = (*current);
				Float3 deltaR = p->pos - j.pos;

				if (q <= 0.5) {
					p->norm += param->mass * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR / distance / j.dens;
				}

				else if (q <= 1) {
					p->norm += param->mass * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR / distance / j.dens;
				}
			}
		}
		}

		p->norm = param->h * p->norm;
	}
}

static void Cpu_generateHashTable(Particle* particles, uint* dParticleIndex, uint* dCellIndex, Param* param) {
	for (int index = 0; index < param->num_particles; index++) {
		// Compute the cell index
		uint hash = computeCellHash(computeCellPosition(particles[index].pos, param), param);
		if (hash >= param->cells_total)
			return;
		dParticleIndex[index] = index;
		dCellIndex[index] = hash;
	}
}

static void Cpu_sort_particles(uint *dHash, uint *dIndex, int num_particle) {
	if (num_particle == 0)
		return;

	struct Interval
	{
		int left = 0;
		int right = 0;

		Interval(int newLeft, int newRight) {
			left = newLeft;
			right = newRight;
		}

		int getMiddle() {
			return (left + right) / 2;
		}
	};

	std::vector<Interval> stack;
	std::vector<uint> smallHashStack;
	std::vector<uint> bigHashStack;
	std::vector<uint> smallIndexStack;
	std::vector<uint> bigIndexStack;

	Interval interval(0, num_particle);
	Interval *current;

	stack.push_back(interval);

	while (stack.size() != 0) {
		current = &stack[stack.size() - 1];
		if (current->right - current->left <= 1) {
			stack.pop_back();
			continue;
		}
		int pivot_index = current->getMiddle();
		uint pivot = dHash[pivot_index];
		uint pivot_dIndex = dIndex[pivot_index];
		for (int i = current->left; i < current->right; i++) {
			if (i == pivot_index)
				continue;
			if (dHash[i] < pivot) {
				smallHashStack.push_back(dHash[i]);
				smallIndexStack.push_back(dIndex[i]);
			}
			else {
				bigHashStack.push_back(dHash[i]);
				bigIndexStack.push_back(dIndex[i]);
			}
		}
		int j = current->left;
		for (int k = 0; k < smallHashStack.size(); k++) {
			dHash[j] = smallHashStack[k];
			dIndex[j] = smallIndexStack[k];
			j++;
		}
		dHash[j] = pivot;
		dIndex[j] = pivot_dIndex;
		j++;
		for (int k = 0; k < bigHashStack.size(); k++) {
			dHash[j] = bigHashStack[k];
			dIndex[j] = bigIndexStack[k];
			j++;
		}
		smallHashStack.clear();
		smallIndexStack.clear();
		bigHashStack.clear();
		bigIndexStack.clear();
		stack.pop_back();
		Interval leftSegment(current->left, pivot_index);
		Interval rightSegment(pivot_index, current->right);
		stack.push_back(leftSegment);
		stack.push_back(rightSegment);
	}
}

static void Cpu_DFSPHComputeForces(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex) {
	// Suppose now we've got the external forces
	for (int index = 0; index < param->num_particles; index++) {
		Uint3 neighborPos;
		Uint3 cellPos = computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		p->acc.x = p->acc.z = 0;
		p->acc.y = GRAVITY;
		uint count = 0;

		ITERATE_NEIGHBOR{
		neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		int hash = computeCellHash(neighborPos, param);
		if (hash<0 || hash >= param->cells_total)
			continue;
		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *current = &particles[dParticleIndex[count]];
				float distance = current->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;
				// To optimize the calculation process
				Particle j = (*current);
				Float3 delta_v = p->vel - j.vel;

				if (q <= 0.5) {
					p->acc -= param->vicosity_coff * (param->mass / j.dens) * delta_v * param->spline_coff * (6 * pow(q, 3) - 6 * pow(q, 2) + 1) / param->timeStep;
				}

				else if (q <= 1) {
					p->acc -= param->vicosity_coff * (param->mass / j.dens) * delta_v * param->spline_coff * 2 * pow(1 - q, 3) / param->timeStep;
				}
			}
		}
		}
	}
}

static void Cpu_DFSPHComputeSurfaceTensionForce(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd) {
	for (int index = 0; index < param->num_particles; index++) {
		int hash;
		Uint3 neighborPos;
		Uint3 cellPos = computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		uint count = 0;

		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = computeCellHash(neighborPos, param);
		if (hash < 0 || hash >= param->cells_total)
			continue;
		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *current = &particles[dParticleIndex[count]];
				float distance = current->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;
				// To optimize the calculation process
				Particle j = (*current);
				float K_ij = 2.0 * param->rest_density / (p->dens + j.dens);
				Float3 temp;
				temp.x = temp.y = temp.z = 0.0f;
				Float3 deltaR = p->pos - j.pos;

				if (deltaR.NormSquare() > 1.0e-9) {
					deltaR = deltaR * (1.0 / deltaR.Norm());
					if (q > 0.5)
						temp -= param->surf_tens_coff * param->mass * deltaR * param->cohesion_coff * pow(param->h - distance, 3) * pow(distance, 3);
					else
						temp -= param->surf_tens_coff * param->mass * deltaR * param->cohesion_coff * 2.0f * pow(param->h - distance, 3) * pow(distance, 3) - param->cohesion_term;
				}

				temp -= param->surf_tens_coff * param->h * (p->norm - j.norm);

				p->acc += K_ij * temp;
			}
		}
		}
	}
}

static void Cpu_DFSPHComputeVelocityScalar(Particle* particles, Param* param) {
	for (int index = 0; index < param->num_particles; index++) {
		if (index >= param->num_particles)
			return;
		Particle *p = &particles[index];
		p->vel_scalar = p->vel.Norm();
	}
}

static void Cpu_DFSPHUpdateTimeStep(Particle* particles, Param* param) {
	float max_vel = particles[0].vel_scalar;
	for (int i = 1; i < param->num_particles; i++) {
		if (particles[i].vel_scalar > max_vel)
			max_vel = particles[i].vel_scalar;
	}

	// Use the middle value
	param->timeStep = 0.5 * 0.4 * 2 * param->radius / (max_vel + 1e-6);

	if (param->timeStep > 0.005)
		param->timeStep = 0.005;

	else if (param->timeStep < 0.0001)
		param->timeStep = 0.0001;

}

static void Cpu_DFSPHPredictVelocity(Particle* particles, Param* param) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= param->num_particles)
		return;
	Particle *p = &particles[index];

	p->vel.x += param->timeStep * p->acc.x;
	p->vel.y += param->timeStep * p->acc.y;
	p->vel.z += param->timeStep * p->acc.z;

}

static void Cpu_DFSPHPredictDensAndVelocity(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood) {
	for (int index = 0; index < param->num_particles; index++) {
		uint hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);

		p->predict_dens = 0;

		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = computeCellHash(neighborPos, param);
		if (hash >= param->cells_total)
			continue;

		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;
				Float3 deltaV = p->vel - j->vel;

				// Compute Density
				if (q <= 0.5)
					p->predict_dens += param->mass * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR.Dot(deltaV) / distance;
				else if (q <= 1)
					p->predict_dens += param->mass * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR.Dot(deltaV) / distance;
			}
		}
		}

		p->predict_dens = p->dens + param->timeStep * p->predict_dens;

		if (p->predict_dens < param->rest_density)
			p->predict_dens = param->rest_density;

		// CUDA calculation accruary issue
		float dens_err = p->predict_dens - param->rest_density;
		if (dens_err > 1e-6)
			param->avg_dens += dens_err;
	}
}

static void Cpu_DFSPHDensitySolverPart3(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood) {

	for (int index = 0; index < param->num_particles; index++) {
		uint hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);

		float ki = (p->predict_dens - param->rest_density) * p->alpha / (param->timeStep*param->timeStep);


		float kj = 0;
		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = computeCellHash(neighborPos, param);
		if (hash >= param->cells_total)
			continue;
		if (dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;
				kj = (j->predict_dens - param->rest_density) * j->alpha / (param->timeStep*param->timeStep);

				// Prevent instability
				float tk = ki + kj;
				if (-1e-6 >= tk || tk >= 1e-6) {
					if (q <= 0.5) {
						p->vel += param->timeStep * tk * param->mass * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR / distance;
					}
					else if (q <= 1) {
						p->vel += param->timeStep * tk * param->mass * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR / distance;
					}
				}
			}
		}
		}
	}
}

static void Cpu_DFSPHDensitySolverPart2(Param* param, int *isGood) {
	if (-0.1 < param->avg_dens && param->avg_dens < 0.1) {
		*isGood = 1;
		return;
	}
	param->avg_dens = param->avg_dens / param->num_particles;
	if (-0.1 < param->avg_dens && param->avg_dens < 0.1)
		*isGood = 1;
}

static void Cpu_find_start_end_kernel(uint *dStart, uint *dEnd, uint *dCellIndex, uint *dParticleIndex, uint num_particle)
{
	// For each index in the dParticleIndex
	for (int index = 0; index < num_particle; index++) {
		// If index == 0
		if (index == 0) {
			// Then we check if we have any particle
			if (dCellIndex[0] == 0xffffffff && dCellIndex[1] == 0xffffffff)
				return;
			else if (dCellIndex[1] == 0xffffffff) {
				dStart[dCellIndex[0]] = index;
				dEnd[dCellIndex[0]] = index;
				return;
			}
			else if (dCellIndex[index] == dCellIndex[index + 1])
				dStart[dCellIndex[index]] = index;
			else {
				dStart[dCellIndex[index]] = dEnd[dCellIndex[index]] = index;
				dStart[dCellIndex[index + 1]] = index + 1;
			}
		}

		else if (index == num_particle - 1) {
			if (dCellIndex[index] == 0xffffffff && dCellIndex[index - 1] == 0xffffffff)
				return;
			else if (dCellIndex[index] == 0xffffffff) {
				dEnd[dCellIndex[index - 1]] = index - 1;
			}
			else if (dCellIndex[index] == dCellIndex[index - 1])
				dEnd[dCellIndex[index]] = index;
			else {
				dStart[dCellIndex[index]] = index;
				dEnd[dCellIndex[index]] = index;
			}
		}

		else if (dCellIndex[index] == dCellIndex[index + 1]) {
			return;
		}

		else {
			if (dCellIndex[index] != 0xffffffff && dCellIndex[index + 1] == 0xffffffff) {
				dEnd[dCellIndex[index]] = index;
			}
			else {
				dEnd[dCellIndex[index]] = index;
				dStart[dCellIndex[index + 1]] = index + 1;
			}
		}
	}
}

static void Cpu_DFSPHCommputeDensityAndFactorAlpha(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd) {
	for (int index = 0; index < param->num_particles; index++)
	{
		uint hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		p->dens = param->mass * param->spline_coff;
		p->alpha = 0;
		Float3 temp, temp2;
		temp.x = temp.y = temp.z = 0;
		temp2.x = temp2.y = temp2.z = 0;

		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = computeCellHash(neighborPos, param);
		if (hash >= param->cells_total)
			continue;
		if (dStart[hash] >= 0 && dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;

				// Compute Density
				if (q <= 0.5)
					p->dens += param->mass * param->spline_coff * (6 * pow(q, 3) - 6 * pow(q, 2) + 1);
				else if (q <= 1)
					p->dens += param->mass * param->spline_coff * 2 * pow(1 - q, 3);

				// Compute Factor Alpha
				if (q <= 0.5) {
					temp2 = param->mass * param->grad_spline_coff * q * (3.0f*q - 2.0f) * deltaR / distance;
					p->alpha += temp2.NormSquare();
					temp += temp2;
				}

				else if (q <= 1) {
					temp2 = param->mass * param->grad_spline_coff * (-1) * pow(1.0f - q, 2) * deltaR / distance;
					p->alpha += temp2.NormSquare();
					temp += temp2;
				}
			}
		}
		}


		p->alpha += temp.NormSquare();

		if (p->alpha < 1e-6)
			p->alpha = 1e-6;

		p->alpha = -1.0f / p->alpha;

	}

}

static void Cpu_DFSPHDivergenceSolver1(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood) {

	for (int index = 0; index < param->num_particles; index++) {
		int hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);
		p->grad_dens = 0.0f;
		int neighbor_count = 0;

		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = computeCellHash(neighborPos, param);
		if (hash < 0 || hash >= param->cells_total)
			continue;
		if (dStart[hash] >= 0 && dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;
				Float3 deltaV = p->vel - j->vel;

				// Compute Density
				if (q <= 0.5)
					p->grad_dens += param->mass  * param->grad_spline_coff * q * (3.0f*q - 2.0f) * (deltaR.Dot(deltaV)) / distance;
				else if (q <= 1)
					p->grad_dens += param->mass * param->grad_spline_coff * (-1) * pow(1 - q, 2) * (deltaR.Dot(deltaV)) / distance;
			}
		}

		if (dBoundaryStart[hash] < 0 || dBoundaryStart[hash] >= param->num_boundary_particles)
			continue;
		for (count = dBoundaryStart[hash]; count <= dBoundaryEnd[hash]; count++) {
			Particle *j = &dBoundaryParticles[dBoundaryParticleIndex[count]];
			float distance = j->pos.Dist(p->pos);
			float q = distance / param->h;
			if (q > 1 || q <= 0)
				continue;

			Float3 deltaR = p->pos - j->pos;
			Float3 deltaV = p->vel - j->vel;
			neighbor_count++;

			// Compute Density Gradient
			if (q <= 0.5)
				p->grad_dens += j->Psi  * param->grad_spline_coff * q * (3.0f*q - 2.0f) * (deltaR.Dot(deltaV)) / distance;
			else if (q <= 1)
				p->grad_dens += j->Psi * param->grad_spline_coff * (-1) * pow(1 - q, 2) *(deltaV.Dot(deltaR)) / distance;
		}
		}


			if (p->grad_dens < 0)
				p->grad_dens = 0;
		if (neighbor_count < 20)
			p->grad_dens = 0;

		// CUDA calculation accruary issue
		if (p->grad_dens > 1e-6)
			param->avg_grad_dens += p->grad_dens;
	}
}

static void Cpu_DFSPHDivergenceSolver3(Particle* particles, Param* param, uint* dStart, uint* dEnd, uint* dParticleIndex,
	Particle* dBoundaryParticles, uint* dBoundaryParticleIndex, uint* dBoundaryCellIndex, uint* dBoundaryStart, uint* dBoundaryEnd, int *isGood) {
	for (int index = 0; index < param->num_particles; index++) {
		int hash;
		uint count = 0;
		Uint3 neighborPos;
		Uint3 cellPos = computeCellPosition(particles[index].pos, param);
		Particle *p = &(particles[index]);

		float ki = p->grad_dens * p->alpha / param->timeStep;
		float kj = 0;
		ITERATE_NEIGHBOR{
			neighborPos.x = cellPos.x + x;
		neighborPos.y = cellPos.y + y;
		neighborPos.z = cellPos.z + z;
		hash = computeCellHash(neighborPos, param);
		if (hash < 0 || hash >= param->cells_total)
			continue;
		if (dStart[hash] >= 0 && dStart[hash] < param->num_particles) {
			for (count = dStart[hash]; count <= dEnd[hash]; count++) {
				Particle *j = &particles[dParticleIndex[count]];
				float distance = j->pos.Dist(p->pos);
				float q = distance / param->h;
				if (q > 1 || q <= 0)
					continue;

				Float3 deltaR = p->pos - j->pos;
				kj = j->grad_dens * j->alpha / param->timeStep;

				// Compute Density
				float tk = ki + kj;
				if (tk < -1e-6 || tk > 1e-6) {
					if (q <= 0.5) {
						p->vel += param->timeStep * param->mass * param->grad_spline_coff*(ki + kj) * q * (3.0f*q - 2.0f) * deltaR / distance;
					}
					else if (q <= 1) {
						p->vel += param->timeStep*param->mass * param->grad_spline_coff * (ki + kj) * (-1) * pow(1 - q, 2) * deltaR / distance;
					}
				}
			}
		}

		if (dBoundaryStart[hash] < 0 || dBoundaryStart[hash] >= param->num_boundary_particles)
			continue;
		for (count = dBoundaryStart[hash]; count <= dBoundaryEnd[hash]; count++) {
			Particle *j = &dBoundaryParticles[dBoundaryParticleIndex[count]];
			float distance = j->pos.Dist(p->pos);
			float q = distance / param->h;
			if (q > 1 || q <= 0)
				continue;

			Float3 deltaR = p->pos - j->pos;

			// Compute Density
			if (ki < -1e-6 || ki > 1e-6) {
				if (q <= 0.5) {
					p->vel += param->timeStep * j->Psi * param->grad_spline_coff * (ki)* q * (3.0f*q - 2.0f) * deltaR / distance;
				}
				else if (q <= 1) {
					p->vel += param->timeStep * j->Psi * param->grad_spline_coff * (ki) * (-1) * pow(1 - q, 2) * deltaR / distance;
				}
			}
		}
		}
	}
}

static void Cpu_DFSPHDivergenceSolver2(Param* param, int *isGood) {
	param->avg_grad_dens = param->avg_grad_dens / param->num_particles;
	if (param->avg_grad_dens < (1.0f / param->timeStep))
		*isGood = 1;
}