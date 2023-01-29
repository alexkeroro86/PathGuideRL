#pragma once

#include "rng.cuh"

// alias method
__device__ inline void sample_cdf(const rng& rng, curandState* rand_state, const float* pdf, int len, int& idx, float* norm_pdf, float eta)
{
	// init
	float* cdf = new float[len + 1];
	cdf[0] = 0.f;
	for (int i = 1; i < len + 1; ++i) {
		cdf[i] = eta + pdf[i - 1] + cdf[i - 1];
	}
	for (int i = 1; i < len + 1; ++i) {
		cdf[i] /= cdf[len];
		norm_pdf[i - 1] = cdf[i] - cdf[i - 1];
	}

	// rand
	float r = rng.generate_random(rand_state);
	for (int i = 0; i < len; ++i) {
		if (cdf[i] <= r && r < cdf[i + 1]) {
			idx = i;
			break;
		}
	}

	// free pointer
	delete[] cdf;
}

