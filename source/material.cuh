#pragma once

#include "ray.cuh"
#include "pdf.cuh"
#include "surf_tex.cuh"
#include "hitable.cuh"
#include "constant.h"

struct scatter_record
{
	ray specular_ray;
	bool is_specular;
	bool is_area_light;
	vec3 attenuation;
	pdf* pdf_ptr;
};

class material
{
public:
	// L_i in rendering equation
	__device__ virtual bool scatter(const ray& r_in, const hit_record& hrec, scatter_record& srec) { return false; }
	// L_e in rendering equation
	__device__ virtual vec3 emitted(const ray& r_in, const hit_record& hrec, float u, float v, const vec3& p) { return vec3(0.f); }
	// cos_theta in rendering equation
	__device__ virtual float scattering_pdf(const ray& r_in, const hit_record& hrec, const ray& scattered) { return false; }
};

class lambertian : public material
{
public:
	__device__ lambertian(surf_tex* a) : albedo(a) {}

	__device__ bool scatter(const ray& r_in, const hit_record& hrec, scatter_record& srec) override
	{
		srec.is_specular = false;
		srec.attenuation = albedo->value(hrec.u, hrec.v, hrec.p);
		// cosine density for brdf sampling
		srec.pdf_ptr = new cosine_pdf(hrec.normal);
		return true;
	}
	__device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) override
	{
		float cosine = dot(rec.normal, unit_vector(scattered.d()));
		if (cosine < 0.f) cosine = 0.f;
		return cosine / PI;
	}
private:
	surf_tex* albedo;
};

class area_light : public material
{
public:
	__device__ area_light(surf_tex* a) : emit(a) {}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& hrec, scatter_record& srec) override
	{
		srec.is_area_light = true;
		return false;
	}

	__device__ virtual vec3 emitted(const ray& r_in, const hit_record& hrec, float u, float v, const vec3& p) override
	{
		// avoid backface
		if (dot(hrec.normal, r_in.d()) < 0.f) {
			return emit->value(u, v, p);
		}
		else {
			return vec3(0.f);
		}
	}

private:
	surf_tex* emit;
};

