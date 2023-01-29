#pragma once

#include "vec3.cuh"

class onb
{
public:
	__device__ onb() {}

	// operator
	__device__ inline vec3 operator[](int i) const { return axis[i]; }

	// access function
	__device__ vec3 u() const { return axis[0]; }
	__device__ vec3 v() const { return axis[1]; }
	__device__ vec3 w() const { return axis[2]; }

	// member function
	__device__ inline vec3 local(float a, float b, float c) const { return a * u() + b * v() + c * w(); }
	__device__ inline vec3 local(const vec3& a) const { return a.x() * u() + a.y() * v() + a.z() * w(); }
	__device__ void build_from_w(const vec3& n);

private:
	vec3 axis[3];
};

__device__ void onb::build_from_w(const vec3 &n)
{
	axis[2] = unit_vector(n);
	// avoid parallel to w
	vec3 a;
	if (fabs(w().x()) > 0.9f) {
		a = vec3(0.f, 1.f, 0.f);
	}
	else {
		a = vec3(1.f, 0.f, 0.f);
	}
	axis[1] = unit_vector(cross(w(), a));
	axis[0] = cross(w(), v());
}

