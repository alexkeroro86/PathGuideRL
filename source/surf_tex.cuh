#pragma once

#include "vec3.cuh"

class surf_tex
{
public:
	__device__ surf_tex() {}
	__device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public surf_tex
{
public:
	__device__ constant_texture() : albedo(0.f) {}
	__device__ constant_texture(vec3 c) : albedo(c) {}

	__device__ virtual vec3 value(float u, float v, const vec3& p) const
	{
		return albedo;
	}

private:
	// reflectance of surface
	vec3 albedo;
};

