#pragma once

#include "ray.cuh"
#include "rng.cuh"
#include "aabb.cuh"

struct hit_record
{
	float t;
	float u;
	float v;
	vec3 p;
	vec3 normal;
	class material* mat_ptr;
	class regular_grid* rg_ptr;
	int uid;
};

// light sampling
class hitable
{
public:
	__device__ hitable(int light_uid = -1) : uid(light_uid) {}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) = 0;
	// pdf in area form (NOTE: pdf is multiplied by geometry term in implementation) (second called)
	__device__ virtual float pdf_value(const vec3& hit_point, const vec3& scattered_dir) { return 0.f; }
	// h' in area form (NOTE: use direction, w'=h'-ray_o in implementation)  (first called)
	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state, const vec3& hit_point) { return vec3(1.f, 0.f, 0.f); }
	// bounding box
	__device__ virtual aabb get_bounding_box() { return aabb(); }

	int uid;
};

class flip_normal : public hitable
{
public:
	__device__ flip_normal(hitable* p) : ptr(p) {}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) override
	{
		if (ptr->hit(r, t_min, t_max, hrec)) {
			hrec.normal = -hrec.normal;
			return true;
		}
		else {
			return false;
		}
	}

	__device__ virtual aabb get_bounding_box() override
	{
		return ptr->get_bounding_box();
	}

private:
	hitable* ptr;
};

