#pragma once

#include "vec3.cuh"
#include "ray.cuh"
#include "hitable.cuh"
#include "rng.cuh"
#include "material.cuh"
#include "td_learning.cuh"
#include "q_learning.cuh"

class path_tracer
{
public:
	__device__ path_tracer() : total_ray(0), valid_ray(0) {}

	__device__ vec3 trace_ray(const ray& r, hitable_list* scene, hitable_list* light_shape, float* norm_patch_pdf, float* norm_light_pdf,
					 int depth, const rng& rng, curandState* rand_state);

public:
	// stats
	long total_ray;
	long valid_ray;
};

__device__ vec3 path_tracer::trace_ray(const ray& r, hitable_list* scene, hitable_list* light_shape, float* norm_patch_pdf, float* norm_light_pdf,
							  int depth, const rng& rng, curandState* rand_state)
{
	ray curr_ray = r;
	ray prev_ray = r;
	vec3 curr_attenuation(1.f);
	vec3 emitted;
	regular_grid* rg_ptr = (regular_grid*)scene;
	bool is_light_sampling = false;
	float geometry_term = 0.f;
	ray scattered;
	float pdf_val;
	int i;

	// q-learning
	int idx_currQ = -1, idx_prevQ = -1, idx_prev_patchQ = -1;
	int cell_idxQ = -1, patch_idx = -1;
	float pdf_valQ;

	// td-learning
	int cell_idxTD = -1, light_idx = -1;
	float pdf_valTD;

	// stats
	total_ray += 1;

	for (i = 0; i < depth; ++i) {
		hit_record hrec;
		hrec.rg_ptr = rg_ptr;
		if (scene->hit(curr_ray, 0.001f, CUDART_TWO_TO_126_F, hrec)) {
			scatter_record srec;
			emitted = hrec.mat_ptr->emitted(curr_ray, hrec, hrec.u, hrec.v, hrec.p);
			if (hrec.mat_ptr->scatter(curr_ray, hrec, srec)) {
				if (srec.is_specular) {
					// TODO
					curr_attenuation *= srec.attenuation;
					curr_ray = srec.specular_ray;
				}
				else {
#if USE_RL
					// w_i in estimator
					if (rng.generate_random(rand_state) < 0.5f) {
						is_light_sampling = true;
						// td-learning
						sample_light(rng, rand_state, rg_ptr, hrec.p, cell_idxTD, light_idx, norm_light_pdf);
						scattered = ray(hrec.p, light_shape->list[light_idx]->generate(rng, rand_state, hrec.p));
					}
					else {
						is_light_sampling = false;
						// q-learning
						if (i > 0) {
							idx_prevQ = rg_ptr->find_cell(curr_ray.o()); // same as `cell_idx`
							idx_currQ = rg_ptr->find_cell(hrec.p);
							idx_prev_patchQ = patch_idx;
						}
						vec3 wi = sample_patch(rng, rand_state, rg_ptr, hrec.p, cell_idxQ, patch_idx, norm_patch_pdf);
						if (i > 0) {
							// EXP: avg or max
							float expect = 0.f;
							for (int j = 0; j < Q_DIM * Q_DIM; ++j) {
								float e = norm_patch_pdf[j] * rg_ptr->patch_proba[cell_idxQ][j];
								//expect += rg_ptr->norm_patch_pdf[j] * rg_ptr->patch_proba[cell_idxQ][j];
								if (expect < e) expect = e;
							}
							// EXP: plus or multiply
							//update_q_table(rg_ptr, idx_prev, idx_prev_patch, curr_attenuation + expect);
							update_q_table(rg_ptr, idx_prevQ, idx_prev_patchQ, curr_attenuation * expect);
						}
						scattered = ray(hrec.p, srec.pdf_ptr->uvw.local(wi));
					}

					// pdf in estimator

					// td-learning
					pdf_valTD = 0.f;
					if (is_light_sampling) {
						for (int j = 0; j < light_shape->list_size; ++j) {
							pdf_valTD += norm_light_pdf[j] * light_shape->list[j]->pdf_value(hrec.p, scattered.d());
						}
					}
					else {
						norm_light(rg_ptr, cell_idxQ, norm_light_pdf);
						for (int j = 0; j < light_shape->list_size; ++j) {
							pdf_valTD += norm_light_pdf[j] * light_shape->list[j]->pdf_value(hrec.p, scattered.d());
						}
					}
					// q-learning
					if (!is_light_sampling) {
						pdf_valQ = norm_patch_pdf[patch_idx] * Q_DIM * Q_DIM / (2.f * PI);
					}
					else {
						norm_patch(rg_ptr, cell_idxTD, norm_patch_pdf);
						vec3 dir = scattered.d();
						dir.make_unit_vector();
						if (dot(hrec.normal, dir) < 0.f) {
							pdf_valQ = 0.f;
						}
						else {
							// FIXME: find patch from direction in brute-force way
							float maxi = 0.f;
							for (int j = 0; j < Q_DIM * Q_DIM; ++j) {
								vec3 cen = srec.pdf_ptr->uvw.local(mini_patch_center(rng, j));
								float val = dot(cen, dir);
								if (val > maxi) {
									maxi = val;
									patch_idx = j;
								}
							}
							pdf_valQ = norm_patch_pdf[patch_idx] * Q_DIM * Q_DIM / (2.f * PI);
							//pdf_valQ = Q_DIM * Q_DIM / (2.f * PI);
						}
					}

					// mis
					pdf_val = pdf_valTD * pdf_valTD / (pdf_valTD + pdf_valQ) + pdf_valQ * pdf_valQ / (pdf_valTD + pdf_valQ);
#else
					hitable_pdf light_pdf(light_shape, hrec.p);
					mixture_pdf mis_pdf(&light_pdf, srec.pdf_ptr);
					// w_i in estimator
					scattered = ray(hrec.p, mis_pdf.generate(rng, rand_state));
					is_light_sampling = mis_pdf.use_light_sampling;
					light_idx = light_shape->rnd_idx;
					// pdf in estimator
					pdf_val = mis_pdf.value(scattered.d());
#endif

					delete srec.pdf_ptr;

					// rendering equation
					geometry_term = hrec.mat_ptr->scattering_pdf(curr_ray, hrec, scattered);
					vec3 Li = srec.attenuation * geometry_term / pdf_val;
					curr_attenuation = emitted + curr_attenuation * Li;
					prev_ray = curr_ray;
					curr_ray = scattered;
				}
			}
			else {
				if (srec.is_area_light) {
					vec3 Lo = emitted * curr_attenuation;
					int visibility = is_light_sampling ? (hrec.uid == light_idx) && !(geometry_term == 0.f) : !(geometry_term == 0.f);

#if USE_RL
					// td-learning
					if (is_light_sampling && i > 0) {
						update_td_table(rg_ptr, cell_idxTD, light_idx, visibility * emitted * geometry_term / sqrtf(hrec.t));
					}
					// q-learning
					if (!is_light_sampling && i > 0) {
						// TODO: distance ?
						update_q_table(rg_ptr, cell_idxQ, patch_idx, visibility * emitted * geometry_term);
					}
#endif

					// stats
					valid_ray += visibility;

					return Lo;
				}
			}
		}
		else {
#if USE_RL
			// td-learning
			if (is_light_sampling && i > 0) {
				update_td_table(rg_ptr, cell_idxTD, light_idx, 0.f);
			}
			// q-learning
			if (!is_light_sampling && i > 0) {
				update_q_table(rg_ptr, cell_idxQ, patch_idx, 0.f);
			}
#endif
			// zero-contribution (miss hit)
			return vec3(0.f);
		}
	}

#if USE_RL
	// q-learning
	if (!is_light_sampling && i > 0) {
		update_q_table(rg_ptr, cell_idxQ, patch_idx, 0.f);
	}
#endif
	// zero-contribution (out of limit)
	return vec3(0.f);
}

