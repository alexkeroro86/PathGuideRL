#pragma once

#include "hitable.cuh"
#include "material.cuh"

class tri : public hitable
{
public:
	__device__ tri() : mat_ptr(nullptr) {}
	__device__ tri(vec3 _v0, vec3 _v1, vec3 _v2, material* mp)
		: v0(_v0), v1(_v1), v2(_v2), mat_ptr(mp)
	{
		n = unit_vector(cross(v1 - v0, v2 - v0));
	}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) override
	{
		return geometric_hit(r, t_min, t_max, hrec);
	}

	__device__ virtual aabb get_bounding_box() override
	{
		float delta = 0.000001f;

		return (aabb(fminf(fminf(v0.x(), v1.x()), v2.x()) - delta, fmaxf(fmaxf(v0.x(), v1.x()), v2.x()) + delta,
					 fminf(fminf(v0.y(), v1.y()), v2.y()) - delta, fmaxf(fmaxf(v0.y(), v1.y()), v2.y()) + delta,
					 fminf(fminf(v0.z(), v1.z()), v2.z()) - delta, fmaxf(fmaxf(v0.z(), v1.z()), v2.z()) + delta));
	}

private:
	// different intersection algorithm
	__device__ inline bool geometric_hit(const ray& r, float t_min, float t_max, hit_record& hrec)
	{
		float ndotv = dot(n, r.d());
		if (fabsf(ndotv) < 0.000001f) {
			return false;
		}
		float d = -dot(n, v0);
		float t = -(dot(n, r.o()) + d) / ndotv;

		if (t < t_min || t > t_max) {
			return false;
		}

		hrec.p = r.point_at_parameter(t);

		// testing edge-0
		vec3 C;
		vec3 e0 = v1 - v0;
		vec3 vp0 = hrec.p - v0;
		C = cross(e0, vp0);
		if (dot(n, C) < 0.f) return false;

		// testing edge-1
		vec3 e1 = v2 - v1;
		vec3 vp1 = hrec.p - v1;
		C = cross(e1, vp1);
		if (dot(n, C) < 0.f) return false;

		// testing edge-2
		vec3 e2 = v0 - v2;
		vec3 vp2 = hrec.p - v2;
		C = cross(e2, vp2);
		if (dot(n, C) < 0.f) return false;

		hrec.t = t;
		hrec.normal = n;
		hrec.mat_ptr = mat_ptr;
		hrec.u = 0.f;
		hrec.v = 0.f;

		return true;
	}
	__device__ inline bool barycentric_coord_hit(const ray& r, float t_min, float t_max, hit_record& hrec)
	{
		float a = v0.x() - v1.x(), b = v0.x() - v2.x(), c = r.d().x(), d = v0.x() - r.o().x();
		float e = v0.y() - v1.y(), f = v0.y() - v2.y(), g = r.d().y(), h = v0.y() - r.o().y();
		float i = v0.z() - v1.z(), j = v0.z() - v2.z(), k = r.d().z(), l = v0.z() - r.o().z();

		float m = f * k - g * j, n2 = h * k - g * l, p = f * l - h * j;
		float q = g * i - e * k, s = e * j - f * i;

		float inv_denom = 1.f / (a * m + b * q + c * s);

		float e1 = d * m - b * n2 - c * p;
		float beta = e1 * inv_denom;

		// early-out
		if (beta < 0.f) return false;

		float r2 = e * l - h * i;
		float e2 = a * n2 + d * q + c * r2;
		float gamma = e2 * inv_denom;

		// early-out
		if (gamma < 0.f) return false;

		// early-out
		if (beta + gamma > 1.f) return false;

		float e3 = a * p - b * r2 + d * s;
		float t = e3 * inv_denom;

		// out-of-range
		if (t < t_min || t > t_max) return false;

		hrec.t = t;
		hrec.normal = n;
		hrec.p = r.point_at_parameter(t);
		hrec.mat_ptr = mat_ptr;
		hrec.u = 0.f;
		hrec.v = 0.f;

		return true;
	}

private:
	// v0, v1, v2 in counterclockwise ordering
	vec3 v0, v1, v2;
	vec3 n;
	material* mat_ptr;
};

