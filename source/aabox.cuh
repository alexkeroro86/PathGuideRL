#pragma once

#include "hitable_list.cuh"
#include "aarect.cuh"

class aabox : public hitable
{
public:
	__device__ aabox() {}
	__device__ aabox(const vec3& _p0, const vec3& _p1, material* m) : p0(_p0), p1(_p1)
	{
		hitable **list = new hitable*[6];
		list[0] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), m);
		list[1] = new flip_normal(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), m));
		list[2] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), m);
		list[3] = new flip_normal(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), m));
		list[4] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), m);
		list[5] = new flip_normal(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), m));
		list_ptr = new hitable_list(list, 6);
	}

	// free pointer
	__device__ ~aabox()
	{
		delete list_ptr;
	}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) override
	{
		return list_ptr->hit(r, t_min, t_max, hrec);
	}

	__device__ virtual aabb get_bounding_box() override
	{
		return aabb(p0, p1);
	}

private:
	vec3 p0, p1;
	hitable_list* list_ptr;
};

