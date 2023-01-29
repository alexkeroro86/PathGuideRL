#pragma once

#include "vec3.cuh"
#include "constant.h"

class rng
{
public:
	__device__ rng() {}

	// a random number in [0, 1)
	__device__ virtual float generate_random(curandState* rand_state) const = 0;
	// a random point in [0, 1) * [0, 1)
	__device__ virtual void random_square_point(curandState* rand_state, float& r1, float& r2) const = 0;
	// map point to unit sphere
	__device__ virtual vec3 square_to_sphere(float r1, float r2, float rad, float dis) const = 0;
	// map point to cosine weigthed distribution
	__device__ virtual vec3 square_to_hemisphere(float r1, float r2, float exp) const = 0;
};

class cuda_rng : public rng
{
public:
	__device__ virtual float generate_random(curandState* rand_state) const override
	{
		return curand_uniform(rand_state);
	}
	__device__ virtual void random_square_point(curandState* rand_state, float& r1, float& r2) const override
	{
		r1 = generate_random(rand_state);
		r2 = generate_random(rand_state);
	}
	__device__ virtual vec3 square_to_sphere(float r1, float r2, float rad, float squared_dis) const override
	{
		float z = 1.f + r2 * (sqrtf(1 - rad * rad / squared_dis) - 1.f);
		float phi = 2.f * PI * r1;
		float x = cosf(phi) * sqrtf(1.f - z * z);
		float y = sinf(phi) * sqrtf(1.f - z * z);
		return vec3(x, y, z);
	}
	__device__ virtual vec3 square_to_hemisphere(float r1, float r2, float exp) const override
	{
		float cos_phi = cosf(2.f * PI * r1);
		float sin_phi = sinf(2.f * PI * r1);
		float cos_theta = powf((1.f - r2), 1.f / (exp + 1.f));
		float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
		float pu = sin_theta * cos_phi;
		float pv = sin_theta * sin_phi;
		float pw = cos_theta;
		return vec3(pu, pv, pw);
	}
};

