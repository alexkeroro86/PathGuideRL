#pragma once

#include "rng.cuh"
#include "constant.h"
#include "regular_grid.cuh"
#include "rl_helper.cuh"

__device__ inline vec3 sample_mini_patch(const rng& rng, curandState* rand_state, int patch_idx)
{
	float r1, r2;
	rng.random_square_point(rand_state, r1, r2);
	int row = patch_idx % Q_DIM;
	int col = patch_idx / Q_DIM;
	float inv_q = 1.f / Q_DIM;
	// EXP: cosine(1) or uniform(0)
	return rng.square_to_hemisphere((row + r1) * inv_q, (col + r2) * inv_q, 0.f);
}

__device__ inline vec3 mini_patch_center(const rng& rng, int patch_idx)
{
	float r1 = 0.5f, r2 = 0.5f;
	int row = patch_idx % Q_DIM;
	int col = patch_idx / Q_DIM;
	float inv_q = 1.f / Q_DIM;
	// EXP: cosine(1) or uniform(0)
	return rng.square_to_hemisphere((row + r1) * inv_q, (col + r2) * inv_q, 0.f);
}

__device__ inline void update_q_table(regular_grid* rg_ptr, int cell_idx, int patch_idx, vec3 val)
{
	// FIXME: VISIBILITY IS INTEGRATED INTO PDF WHICH CAUSES DEVIDED BY ZERO
	if (isnan(val.max()) || isinf(val.max())) val = vec3(0.f);

	float pr = rg_ptr->patch_proba[cell_idx][patch_idx];
	// EXP: patch or cell
	float eta = ETA * 1.f / (1.f + float(rg_ptr->patch_visit[cell_idx][patch_idx]));
	//float eta = ETA * 1.f / (1.f + float(rg_ptr->cell_visit[cell_idx]));
	// atomic operation
	atomicExch(&rg_ptr->patch_proba[cell_idx][patch_idx], (1.f - eta) * pr + eta * val.max());
	atomicAdd(&rg_ptr->patch_visit[cell_idx][patch_idx], 1);
}

__device__ inline vec3 sample_patch(const rng& rng, curandState* rand_state, regular_grid* rg_ptr, const vec3& h, int& cell_idx, int& patch_idx, float* norm_pdf)
{
	cell_idx = rg_ptr->find_cell(h);
	sample_cdf(rng, rand_state, rg_ptr->patch_proba[cell_idx], Q_DIM * Q_DIM, patch_idx, norm_pdf, 0.f);
	return sample_mini_patch(rng, rand_state, patch_idx);
}

__device__ inline void norm_patch(regular_grid* rg_ptr, int cell_idx, float* norm_pdf)
{
	float eta = 0.f;
	float* cdf = new float[Q_DIM * Q_DIM + 1];
	cdf[0] = 0.f;
	for (int i = 1; i < Q_DIM * Q_DIM + 1; ++i) {
		cdf[i] = eta + rg_ptr->patch_proba[cell_idx][i - 1] + cdf[i - 1];
	}
	for (int i = 1; i < Q_DIM * Q_DIM + 1; ++i) {
		cdf[i] /= cdf[Q_DIM * Q_DIM];
		norm_pdf[i - 1] = cdf[i] - cdf[i - 1];
	}
	delete[] cdf;
}

