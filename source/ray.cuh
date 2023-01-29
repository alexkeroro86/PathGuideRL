#pragma once

#include "vec3.cuh"

// formula in point = org + t * dir
class ray
{
public:
	__device__ ray() {}
	__device__ ray(const vec3& o, const vec3& d) : org(o), dir(d) {}

	__device__ vec3 o() const { return org; }
	__device__ vec3 d() const { return dir; }
	__device__ vec3 point_at_parameter(float t) const { return org + t * dir; }

private:
	vec3 org;
	vec3 dir;
};

