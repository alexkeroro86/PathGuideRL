#pragma once

#include "ray.cuh"

class aabb
{
public:
	__device__ aabb() {}
	__device__ aabb(float ax, float bx, float ay, float by, float az, float bz) : pmin(ax, ay, az), pmax(bx, by, bz) {}
	__device__ aabb(vec3 _pmin, vec3 _pmax) : pmin(_pmin), pmax(_pmax) {}

	__device__ bool hit(const ray& r, float t_min, float t_max)
	{
		for (int i = 0; i < 3; ++i) {
			float inv_d = 1.f / r.d()[i];
			float t0 = fminf((pmin[i] - r.o()[i]) * inv_d,
				(pmax[i] - r.o()[i]) * inv_d);
			float t1 = fmaxf((pmin[i] - r.o()[i]) * inv_d,
				(pmax[i] - r.o()[i]) * inv_d);
			t_min = fmaxf(t0, t_min);
			t_max = fminf(t1, t_max);
			if (t_max < t_min) {
				return false;
			}
		}
		return true;
	}

	__device__ inline bool inside(const vec3& p)
	{
		return (p.x() > pmin.x() && p.x() < pmax.x()) &&
			   (p.y() > pmin.y() && p.y() < pmax.y()) &&
			   (p.z() > pmin.z() && p.z() < pmax.z());
	}

	vec3 pmin, pmax;
};

