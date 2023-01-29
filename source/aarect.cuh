#pragma once

#include "hitable.cuh"
#include "material.cuh"
#include "constant.h"

class xz_rect : public hitable
{
public:
	__device__ xz_rect() : hitable() {}
	__device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _y, material* m, int light_uid = -1)
		: x0(_x0), x1(_x1), z0(_z0), z1(_z1), y(_y), mat_ptr(m), hitable(light_uid) {}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) override;
	__device__ virtual float pdf_value(const vec3& hit_point, const vec3& scattered_dir) override;
	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state, const vec3& hit_point) override;
	__device__ virtual aabb get_bounding_box() override;

private:
	float x0, x1, z0, z1, y;
	material* mat_ptr;
};

__device__ inline bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& hrec)
{
	float t = (y - r.o().y()) / r.d().y();
	if (t < t_min || t > t_max) {
		return false;
	}
	float x = r.o().x() + t * r.d().x();
	float z = r.o().z() + t * r.d().z();
	if (x < x0 || x > x1 || z < z0 || z > z1) {
		return false;
	}
	hrec.u = (x - x0) / (x1 - x0);
	hrec.v = (z - z0) / (z1 - z0);
	hrec.t = t;
	hrec.mat_ptr = mat_ptr;
	hrec.p = r.point_at_parameter(t);
	hrec.normal = vec3(0.f, 1.f, 0.f);
	hrec.uid = uid;
	return true;
}
__device__ inline float xz_rect::pdf_value(const vec3& hit_point, const vec3& scattered_dir)
{
	hit_record hrec;
	// visibility function
	if (this->hit(ray(hit_point, scattered_dir), 0.001f, CUDART_TWO_TO_126_F, hrec)) {
		float area = (x1 - x0) * (z1 - z0);
		float distance_squared = hrec.t * hrec.t * scattered_dir.squared_length();
		float cosine = fabs(dot(scattered_dir, hrec.normal) / scattered_dir.length());
		return distance_squared / (cosine * area);
	}
	else {
		return 0.f;
	}
}
__device__ inline vec3 xz_rect::generate(const rng& rng, curandState* rand_state, const vec3& hit_point)
{
	float r1, r2;
	rng.random_square_point(rand_state, r1, r2);
	vec3 sample_point = vec3(x0 + r1 * (x1 - x0), y, z0 + r2 * (z1 - z0));
	return sample_point - hit_point;
}
__device__ inline aabb xz_rect::get_bounding_box()
{
	return aabb(vec3(x0, y, z0) - vec3(EPSILON), vec3(x1, y, z1) + vec3(EPSILON));
}

class xy_rect : public hitable
{
public:
	__device__ xy_rect() : hitable() {}
	__device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _z, material* m, int light_uid = -1)
		: x0(_x0), x1(_x1), y0(_y0), y1(_y1), z(_z), mat_ptr(m), hitable(light_uid) {}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) override;
	__device__ virtual float pdf_value(const vec3& hit_point, const vec3& scattered_dir) override;
	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state, const vec3& hit_point) override;
	__device__ virtual aabb get_bounding_box() override;

private:
	float x0, x1, y0, y1, z;
	material* mat_ptr;
};

__device__ inline bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& hrec)
{
	float t = (z - r.o().z()) / r.d().z();
	if (t < t_min || t > t_max) {
		return false;
	}
	float x = r.o().x() + t * r.d().x();
	float y = r.o().y() + t * r.d().y();
	if (x < x0 || x > x1 || y < y0 || y > y1) {
		return false;
	}
	hrec.u = (x - x0) / (x1 - x0);
	hrec.v = (y - y0) / (y1 - y0);
	hrec.t = t;
	hrec.mat_ptr = mat_ptr;
	hrec.p = r.point_at_parameter(t);
	hrec.normal = vec3(0.f, 0.f, 1.f);
	hrec.uid = uid;
	return true;
}
__device__ inline float xy_rect::pdf_value(const vec3& hit_point, const vec3& scattered_dir)
{
	hit_record hrec;
	// visibility function
	if (this->hit(ray(hit_point, scattered_dir), 0.001f, CUDART_TWO_TO_126_F, hrec)) {
		float area = (x1 - x0) * (y1 - y0);
		float distance_squared = hrec.t * hrec.t * scattered_dir.squared_length();
		float cosine = fabs(dot(scattered_dir, hrec.normal) / scattered_dir.length());
		return distance_squared / (cosine * area);
	}
	else {
		return 0.f;
	}
}
__device__ inline vec3 xy_rect::generate(const rng& rng, curandState* rand_state, const vec3& hit_point)
{
	float r1, r2;
	rng.random_square_point(rand_state, r1, r2);
	vec3 sample_point = vec3(x0 + r1 * (x1 - x0), y0 + r2 * (y1 - y0), z);
	return sample_point - hit_point;
}
__device__ inline aabb xy_rect::get_bounding_box()
{
	return aabb(vec3(x0, y0, z) - vec3(EPSILON), vec3(x1, y1, z) + vec3(EPSILON));
}

class yz_rect : public hitable
{
public:
	__device__ yz_rect() : hitable() {}
	__device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _x, material *m, int light_uid = -1)
		: y0(_y0), y1(_y1), z0(_z0), z1(_z1), x(_x), mat_ptr(m), hitable(light_uid) {}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) override;
	__device__ virtual float pdf_value(const vec3& hit_point, const vec3& scattered_dir) override;
	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state, const vec3& hit_point) override;
	__device__ virtual aabb get_bounding_box() override;

private:
	float y0, y1, z0, z1, x;
	material* mat_ptr;
};

__device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& hrec)
{
	float t = (x - r.o().x()) / r.d().x();
	if (t < t_min || t > t_max) {
		return false;
	}
	float y = r.o().y() + t * r.d().y();
	float z = r.o().z() + t * r.d().z();
	if (y < y0 || y > y1 || z < z0 || z > z1) {
		return false;
	}
	hrec.u = (y - y0) / (y1 - y0);
	hrec.v = (z - z0) / (z1 - z0);
	hrec.t = t;
	hrec.mat_ptr = mat_ptr;
	hrec.p = r.point_at_parameter(t);
	hrec.normal = vec3(1.f, 0.f, 0.f);
	hrec.uid = uid;
	return true;
}
__device__ inline float yz_rect::pdf_value(const vec3& hit_point, const vec3& scattered_dir)
{
	hit_record hrec;
	// visibility function
	if (this->hit(ray(hit_point, scattered_dir), 0.001f, CUDART_TWO_TO_126_F, hrec)) {
		float area = (y1 - y0) * (z1 - z0);
		float distance_squared = hrec.t * hrec.t * scattered_dir.squared_length();
		float cosine = fabs(dot(scattered_dir, hrec.normal) / scattered_dir.length());
		return distance_squared / (cosine * area);
	}
	else {
		return 0.f;
	}
}
__device__ inline vec3 yz_rect::generate(const rng& rng, curandState* rand_state, const vec3& hit_point)
{
	float r1, r2;
	rng.random_square_point(rand_state, r1, r2);
	vec3 sample_point = vec3(x, y0 + r1 * (y1 - y0), z0 + r2 * (z1 - z0));
	return sample_point - hit_point;
}
__device__ inline aabb yz_rect::get_bounding_box()
{
	return aabb(vec3(x, y0, z0) - vec3(EPSILON), vec3(x, y1, z1) + vec3(EPSILON));
}

