#pragma once

#include "hitable.cuh"
#include "onb.cuh"
#include "material.cuh"

class sphere : public hitable
{
public:
	__device__ sphere() : hitable() {}
	__device__ sphere(vec3 c, float r, material* m, int light_uid = -1) : cen(c), rad(r), mat_ptr(m), hitable(light_uid) {}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) override;
	__device__ virtual float pdf_value(const vec3& hit_point, const vec3& scattered_dir) override;
	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state, const vec3& hit_point) override;
	__device__ virtual aabb get_bounding_box() override;

	// TODO: texture mapping
	__device__ void get_sphere_uv(const vec3& p, float& u, float& v);

private:
	vec3 cen;
	float rad;
	material *mat_ptr;
};

__device__ void sphere::get_sphere_uv(const vec3& p, float& u, float& v)
{
	float phi = atan2f(p.z(), p.x());
	float theta = asin(p.y());
	u = 1.f - (phi + PI) / (2.f * PI);
	v = (theta + PI / 2.f) / PI;
}

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& hrec)
{
	vec3 dir = r.o() - cen;
	float a = dot(r.d(), r.d());
	float b = dot(dir, r.d());
	float c = dot(dir, dir) - rad * rad;
	float disc = b * b - a * c;
	if (disc > 0) {
		float root_disc = sqrtf(disc);
		float tmp = (-b - root_disc) / a;
		if (tmp < t_max && tmp > t_min) {
			hrec.t = tmp;
			hrec.p = r.point_at_parameter(tmp);
			get_sphere_uv((hrec.p - cen) / rad, hrec.u, hrec.v);
			hrec.normal = (hrec.p - cen) / rad;
			hrec.mat_ptr = mat_ptr;
			hrec.uid = uid;
			return true;
		}
		tmp = (-b + root_disc) / a;
		if (tmp < t_max && tmp > t_min) {
			hrec.t = tmp;
			hrec.p = r.point_at_parameter(tmp);
			get_sphere_uv((hrec.p - cen) / rad, hrec.u, hrec.v);
			hrec.normal = (hrec.p - cen) / rad;
			hrec.mat_ptr = mat_ptr;
			hrec.uid = uid;
			return true;
		}
	}
	return false;
}
__device__ float sphere::pdf_value(const vec3& hit_point, const vec3& scattered_dir)
{
	hit_record hrec;
	// visibility function
	if (this->hit(ray(hit_point, scattered_dir), 0.001f, CUDART_TWO_TO_126_F, hrec)) {
		float cos_theta_max = sqrtf(1.f - rad * rad / (cen - hit_point).squared_length());
		float solid_angle = 2.f * PI * (1.f - cos_theta_max);
		return 1.f / solid_angle;
	}
	else {
		return 0.f;
	}
}

// return a direction from hit_point to sample_point
__device__ vec3 sphere::generate(const rng& rng, curandState* rand_state, const vec3& hit_point)
{
	vec3 dir = cen - hit_point;
	float squared_dis = dir.squared_length();
	onb uvw;
	uvw.build_from_w(dir);
	float r1, r2;
	rng.random_square_point(rand_state, r1, r2);
	return uvw.local(rng.square_to_sphere(r1, r2, rad, squared_dis));
}

__device__ aabb sphere::get_bounding_box()
{
	return aabb(cen - vec3(rad), cen + vec3(rad));
}

