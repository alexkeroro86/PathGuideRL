#pragma once

#include "rl_helper.cuh"
#include "regular_grid.cuh"

// update pdfs (second called)
__device__ inline void update_td_table(regular_grid* rg_ptr, int cell_idx, int light_idx, vec3 val)
{
	// FIXME: VISIBILITY IS INTEGRATED INTO PDF WHICH CAUSES DEVIDED BY ZERO
	if (isnan(val.max()) || isinf(val.max())) val = vec3(0.f);

	float eta = ETA * 1.f / (1.f + float(rg_ptr->light_visit[cell_idx][light_idx]));
	// atomic operation
	atomicExch(&rg_ptr->light_proba[cell_idx][light_idx], (1 - eta) * rg_ptr->light_proba[cell_idx][light_idx] + eta * val.max());
	atomicAdd(&rg_ptr->light_visit[cell_idx][light_idx], 1);
	atomicAdd(&rg_ptr->cell_visit[cell_idx], 1);
}

// like 'generate' (first called)
__device__ inline void sample_light(const rng& rng, curandState* rand_state, regular_grid* rg_ptr, const vec3& p, int& cell_idx, int& light_idx, float* norm_pdf)
{
	cell_idx = rg_ptr->find_cell(p);
	sample_cdf(rng, rand_state, rg_ptr->light_proba[cell_idx], rg_ptr->light_size, light_idx, norm_pdf, BASE_PDF * 1.f / (1.f + float(rg_ptr->cell_visit[cell_idx])));
	if (light_idx == -1) {
		light_idx = 0;
	}
}

__device__ inline void norm_light(regular_grid* rg_ptr, int cell_idx, float* norm_pdf)
{
	float eta = BASE_PDF * 1.f / (1.f + float(rg_ptr->cell_visit[cell_idx]));
	float* cdf = new float[rg_ptr->light_size + 1];
	cdf[0] = 0.f;
	for (int i = 1; i < rg_ptr->light_size + 1; ++i) {
		cdf[i] = eta + rg_ptr->light_proba[cell_idx][i - 1] + cdf[i - 1];
	}
	for (int i = 1; i < rg_ptr->light_size + 1; ++i) {
		cdf[i] /= cdf[rg_ptr->light_size];
		norm_pdf[i - 1] = cdf[i] - cdf[i - 1];
	}
	delete[] cdf;
}

