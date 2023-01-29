#pragma once

#include "hitable.cuh"

class hitable_list : public hitable
{
public:
	__device__ hitable_list() {}
	__device__ hitable_list(hitable** l, int n) : list(l), list_size(n) {}

	// free pointer
	__device__ ~hitable_list()
	{
		for (int i = 0; i < list_size; ++i) {
			delete list[i];
		}
		delete list;
	}
	
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) override;
	// for light sampling, sample one pdf among multiple lights
	__device__ virtual float pdf_value(const vec3& hit_point, const vec3& scattered_dir) override;
	// for light sampling, sample one h' among multiple lights
	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state, const vec3& hit_point) override;

	__device__ void append(hitable* obj_ptr);

	hitable **list;
	int list_size;
	int rnd_idx;
};

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& hrec)
{
	hit_record tmp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;

	// get the nearest object from ray
	for (int i = 0; i < list_size; ++i) {
		if (list[i]->hit(r, t_min, closest_so_far, tmp_rec)) {
			hit_anything = true;
			closest_so_far = tmp_rec.t;
			hrec = tmp_rec;
		}
	}
	return hit_anything;
}

__device__ float hitable_list::pdf_value(const vec3& hit_point, const vec3& scattered_dir)
{
	// uniformly multiple importance sampling weight between multiple lights
	float weight = 1.f / (float)list_size;
	float sum = 0.f;
	for (int i = 0; i < list_size; ++i) {
		sum += weight * list[i]->pdf_value(hit_point, scattered_dir);
	}
	return sum;
}

__device__ vec3 hitable_list::generate(const rng& rng, curandState* rand_state, const vec3& hit_point)
{
	// choose one light as direct rendering
	rnd_idx = (int)floorf(rng.generate_random(rand_state) * list_size);
	if (rnd_idx == list_size) --rnd_idx;
	return list[rnd_idx]->generate(rng, rand_state, hit_point);
}

__device__ void hitable_list::append(hitable* obj_ptr)
{
	hitable** new_list = new hitable*[list_size + 1];
	int i;
	for (i = 0; i < list_size; ++i) {
		new_list[i] = list[i];
	}
	new_list[i] = obj_ptr;

	// free pointer
	delete[] list;

	// update list
	list_size += 1;
	list = new_list;
}

