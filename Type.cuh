#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
typedef unsigned int uint;


struct Float3 {
	float x;
	float y;
	float z;

	__device__
	Float3 operator+(const Float3& a) {
		Float3 newValue;
		newValue.x = x + a.x;
		newValue.y = y + a.y;
		newValue.z = z + a.z;
		return newValue;
	}

	__device__
	Float3 operator+(const float& a) {
		Float3 newValue;
		newValue.x = x + a;
		newValue.y = y + a;
		newValue.z = z + a;
		return newValue;
	}

	__device__
	void operator+=(const float& a) {
		x = x + a;
		y = y + a;
		z = z + a;
	}

	__device__
	void operator+=(const Float3& a) {
		x = x + a.x;
		y = y + a.y;
		z = z + a.z;
	}

	__device__
	Float3 operator-(const Float3& a) {
		Float3 newValue;
		newValue.x = x - a.x;
		newValue.y = y - a.y;
		newValue.z = z - a.z;
		return newValue;
	}

	__device__
	Float3 operator-(const float& a) {
		Float3 newValue;
		newValue.x = x - a;
		newValue.y = y - a;
		newValue.z = z - a;
		return newValue;
	}

	__device__
	void operator-=(const float& a) {
		x = x - a;
		y = y - a;
		z = z - a;
	}

	__device__
	void operator-=(const Float3& a) {
		x = x - a.x;
		y = y - a.y;
		z = z - a.z;
	}

	__device__
	Float3 operator/(const float& a) {
		Float3 newValue;
		newValue.x = x/a;
		newValue.y = y/a;
		newValue.z = z/a;
		return newValue;
	}

	__device__
	Float3 operator/(const Float3& a) {
		Float3 newValue;
		newValue.x = x / a.x;
		newValue.y = y / a.y;
		newValue.z = z / a.z;
		return newValue;
	}

	__device__
	Float3 operator*(const float& a) {
		Float3 newValue;
		newValue.x = x * a;
		newValue.y = y * a;
		newValue.z = z * a;
		return newValue;
	}

	__device__
	Float3 operator*(const Float3& a) {
		Float3 newValue;
		newValue.x = x * a.x;
		newValue.y = y * a.y;
		newValue.z = z * a.z;
		return newValue;
	}

	__device__
	float Norm() {
		return sqrt(x*x + y*y + z*z);
	}

	__device__
	float NormSquare() {
		return x*x + y*y + z*z;
	}

	__device__
	float Dist(Float3 a) {
		return sqrt((x - a.x)*(x - a.x) + (y - a.y)*(y - a.y) + (z - a.z)*(z - a.z));
	}

	__device__
	float Dot(Float3 a) {
		return x*a.x + y*a.y + z*a.z;
	}
};

__device__
inline Float3 operator+(float a, const Float3& b) {
	Float3 newValue;
	newValue.x = b.x + a;
	newValue.y = b.y + a;
	newValue.z = b.z + a;
	return newValue;
}

__device__
inline Float3 operator-(float a, const Float3& b) {
	Float3 newValue;
	newValue.x = b.x - a;
	newValue.y = b.y - a;
	newValue.z = b.z - a;
	return newValue;
}

__device__
inline Float3 operator*(float a, const Float3& b) {
	Float3 newValue;
	newValue.x = b.x * a;
	newValue.y = b.y * a;
	newValue.z = b.z * a;
	return newValue;
}

__device__
inline Float3 operator/(float a, const Float3& b) {
	Float3 newValue;
	newValue.x = b.x / a;
	newValue.y = b.y / a;
	newValue.z = b.z / a;
	return newValue;
}



struct Uint3 {
	uint x;
	uint y;
	uint z;

	Uint3 operator+(const Uint3& a) {
		Uint3 newValue;
		newValue.x = x + a.x;
		newValue.y = y + a.y;
		newValue.z = z + a.z;
		return newValue;
	}
};

class Particle
{
public:
	// Particle's position
	Float3 pos;
	// Paticle's velocity
	Float3 vel;
	// Particle's acceleration
	Float3 acc;
	// Paticle's mass
	float mass;
	// the Gradient of Particle's Color
	Float3 grad_color;
	// the Laplacian of Particle's Color
	float lplc_color;
	// Factor Alpha
	float alpha;
	// Density grad
	float grad_dens;
	// Velocity scalar
	float vel_scalar;
	// Particle diameter
	float radius;
	// Psi
	float Psi;
	// Norm
	Float3 norm;

	// Patcile's mass density
	float dens;
	// Particle's predict mass density
	float predict_dens;
	// Paticle's pressure
	float pres;

	// Forces
	Float3 F_Pressure;
	Float3 F_Viscosity;
	Float3 F_SurfaceTension;
	Float3 F_Graivity;


	// Flag
	bool isObject;
};

struct vertex {
	Float3 pos;
	float val;
	Float3 norm;
};

struct cube {
	Particle* p;
	vertex vertices[8];
};
