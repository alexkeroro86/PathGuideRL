#pragma once

#include "rng.cuh"
#include "onb.cuh"
#include "hitable.cuh"

class pdf
{
public:
	// pdf in spherical form (second called)
	__device__ virtual float value(const vec3& dir) = 0;
	// w' in spherical form (first called)
	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state) = 0;
	__device__ virtual ~pdf() {}

	onb uvw;
};

// brdf sampling
class uniform_pdf : public pdf
{
public:
	__device__ uniform_pdf(const vec3& w) { uvw.build_from_w(w); }

	__device__ virtual float value(const vec3& dir) override
	{
		float cosine = dot(unit_vector(dir), uvw.w());
		if (cosine > 0) {
			return 1.f / (2.f * PI);
		}
		else {
			return 0.f;
		}
	}
	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state) override
	{
		float r1, r2;
		rng.random_square_point(rand_state, r1, r2);
		return uvw.local(rng.square_to_hemisphere(r1, r2, 0.f));
	}
};

class cosine_pdf : public pdf
{
public:
	__device__ cosine_pdf(const vec3& w) { uvw.build_from_w(w); }

	__device__ virtual float value(const vec3& dir) override
	{
		float cosine = dot(unit_vector(dir), uvw.w());
		if (cosine > 0) {
			return cosine / PI;
		}
		else {
			return 0.f;
		}
	}
	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state) override
	{
		float r1, r2;
		rng.random_square_point(rand_state, r1, r2);
		return uvw.local(rng.square_to_hemisphere(r1, r2, 1.f));
	}
};

// light sampling
class hitable_pdf : public pdf
{
public:
	__device__ hitable_pdf(hitable* p, const vec3& h) : ptr(p), hit_point(h) {}

	__device__ virtual float value(const vec3& dir) override
	{
		return ptr->pdf_value(hit_point, dir);
	}

	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state) override
	{
		return ptr->generate(rng, rand_state, hit_point);
	}

private:
	vec3 hit_point;
	hitable* ptr;
};

// multiple importance sampling between light and brfd sampling
class mixture_pdf : public pdf
{
public:
	bool use_light_sampling;

public:
	__device__ mixture_pdf(pdf* p0, pdf* p1) : p{p0, p1} {}

	__device__ virtual float value(const vec3& dir) override
	{
		return balance_heuristic(dir);
	}

	__device__ virtual vec3 generate(const rng& rng, curandState* rand_state) override
	{
		if (rng.generate_random(rand_state) < 0.5f) {
			use_light_sampling = true;
			return p[0]->generate(rng, rand_state);
		}
		else {
			use_light_sampling = false;
			return p[1]->generate(rng, rand_state);
		}
	}

private:
	// different mixing method
	__device__ inline float arithmetic_average(const vec3& dir)
	{
		// arithmetic average weight for mis
		return 0.5f * p[0]->value(dir) + 0.5f * p[1]->value(dir);
	}
	__device__ inline float balance_heuristic(const vec3& dir)
	{
		// balance heuristic weight for mis
		float p0 = p[0]->value(dir);
		float p1 = p[1]->value(dir);
		return p0 * p0 / (p0 + p1) + p1 * p1 / (p0 + p1);
	}

private:
	pdf* p[2];
};

