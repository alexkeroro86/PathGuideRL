#pragma once

#include "ray.cuh"
#include "onb.cuh"
#include "constant.h"

class camera
{
public:
	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect)
		: vfov(vfov), aspect(aspect)
	{
		float theta = vfov * PI / 180.f;
		float half_height = tanf(theta * 0.5f);
		float half_width = aspect * half_height;

		// get local basis on camera
		origin = lookfrom;
		uvw.build_from_w(lookfrom - lookat);
		lower_left_corner = origin - half_width * uvw.u() - half_height * uvw.v() - uvw.w();

		// get local basis on viewplane
		horizontal = 2.f * half_width * uvw.u();
		vertical = 2.f * half_height * uvw.v();
	}

	__device__ ray get_primary_ray(float s, float t)
	{
		return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
	}

	__device__ void update(float DEG2RAD, vec3 lookdir, vec3 lookfrom, vec3 UP)
	{
		float theta = vfov * DEG2RAD;
		float half_height = tanf(theta * 0.5f);
		float half_width = aspect * half_height;

		vec3 forward = unit_vector(-lookdir);
		vec3 right = unit_vector(cross(UP, forward));
		vec3 up = cross(forward, right);

		origin = lookfrom;
		lower_left_corner = origin - right * half_width - up * half_height - forward;
		horizontal = 2.f * half_width * right;
		vertical = 2.f * half_height * up;
	}

private:
	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	onb uvw;
	float vfov;
	float aspect;
};

