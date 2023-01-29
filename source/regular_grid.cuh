#pragma once

#include "hitable_list.cuh"
#include "aabb.cuh"
#include "constant.h"
#include "pdf.cuh"
#include "rng.cuh"

class regular_grid : public hitable_list
{
public:
	__device__ regular_grid() {}
	__device__ regular_grid(hitable** l, int n) : hitable_list(l, n) {}

	__device__ ~regular_grid()
	{
		// free pointer
		delete[] cells;

#if USE_RL
		for (int i = 0; i < light_size; ++i) {
			delete[] light_proba[i];
		}
		delete[] light_proba;

		for (int i = 0; i < nx * ny * nz; ++i) {
			delete[] light_visit[i];
		}
		delete[] light_visit;

		delete[] cell_visit;

		for (int i = 0; i < nx * ny * nz; ++i) {
			delete[] patch_proba[i];
		}
		delete[] patch_proba;

		for (int i = 0; i < nx * ny * nz; ++i) {
			delete[] patch_visit[i];
		}
		delete[] patch_visit;
#endif
	}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& hrec) override;
	__device__ virtual aabb get_bounding_box() override { return bbox; }

	__device__ int find_cell(const vec3& p);
	__device__ void setup_cells(curandState* rand_state, int num_lights);
	__device__ void dump_proba();

private:
	__device__ vec3 find_min_coordinates();
	__device__ vec3 find_max_coordinates();

public:
	// td learning
	int light_size;
	float inv_light_size;
	float** light_proba;
	int** light_visit;
	int* cell_visit;

	// q learning
	float** patch_proba;
	int** patch_visit;

private:
	hitable** cells;
	aabb bbox;
	int nx, ny, nz;

	// bounding box corners
	float x0, y0, z0;
	float x1, y1, z1;
};

__device__ inline vec3 regular_grid::find_min_coordinates()
{
	aabb object_box;
	vec3 pmin(CUDART_TWO_TO_126_F);

	for (int i = 0; i < list_size; ++i) {
		object_box = list[i]->get_bounding_box();

		pmin[0] = fminf(pmin[0], object_box.pmin[0]);
		pmin[1] = fminf(pmin[1], object_box.pmin[1]);
		pmin[2] = fminf(pmin[2], object_box.pmin[2]);
	}
	pmin -= vec3(EPSILON);

	return pmin;
}
__device__ inline vec3 regular_grid::find_max_coordinates()
{
	aabb object_box;
	vec3 pmax(-CUDART_TWO_TO_126_F);

	for (int i = 0; i < list_size; ++i) {
		object_box = list[i]->get_bounding_box();

		pmax[0] = fmaxf(pmax[0], object_box.pmax[0]);
		pmax[1] = fmaxf(pmax[1], object_box.pmax[1]);
		pmax[2] = fmaxf(pmax[2], object_box.pmax[2]);
	}
	pmax += vec3(EPSILON);

	return pmax;
}

__device__ inline int regular_grid::find_cell(const vec3& p)
{
	int ix = CLAMP(roundf((p.x() - x0) / (x1 - x0) * float(nx)), 0, nx - 1);
	int iy = CLAMP(roundf((p.y() - y0) / (y1 - y0) * float(ny)), 0, ny - 1);
	int iz = CLAMP(roundf((p.z() - z0) / (z1 - z0) * float(nz)), 0, nz - 1);
	
	return ix + nx * iy + nx * ny * iz;
}

__device__ inline void regular_grid::setup_cells(curandState* rand_state, int num_lights)
{
	// get bounding box of regular grid
	vec3 pmin = find_min_coordinates();
	vec3 pmax = find_max_coordinates();

	bbox = aabb(pmin, pmax);

	// compute number of cells
	vec3 grid_dim(pmax - pmin);
	float s = powf(grid_dim.x() * grid_dim.y() * grid_dim.x() / float(list_size), 0.3333f);
	nx = int(floorf(GRID_MULTIPLIER * grid_dim.x() / s)) + 1;
	ny = int(floorf(GRID_MULTIPLIER * grid_dim.y() / s)) + 1;
	nz = int(floorf(GRID_MULTIPLIER * grid_dim.z() / s)) + 1;

	// regular grid state
	printf("grid dimension: (%f, %f, %f)\n", grid_dim[0], grid_dim[1], grid_dim[2]);
	printf("cell dimension: (%d, %d, %d)\n", nx, ny, nz);
	printf("number of lights: %d\n", num_lights);
	printf("bounding box: (%f, %f, %f) and (%f, %f, %f)\n\n", pmin[0], pmin[1], pmin[2], pmax[0], pmax[1], pmax[2]);

	// allocate cells
	int num_cells = nx * ny * nz;
	cells = new hitable*[num_cells];
	memset(cells, 0, num_cells * sizeof(hitable*));
	int* counts = new int[num_cells];
	memset(counts, 0, num_cells * sizeof(int));

#if USE_RL
	// allocate proba for td learning
	light_size = num_lights;
	inv_light_size = 1.f / num_lights;
	light_proba = new float*[num_cells];
	for (int i = 0; i < num_cells; ++i) {
		light_proba[i] = new float[num_lights];
		for (int j = 0; j < num_lights; ++j) {
			light_proba[i][j] = 0.f;
		}
	}

	// allocate light visit for td learning
	light_visit = new int*[num_cells];
	for (int i = 0; i < num_cells; ++i) {
		light_visit[i] = new int[num_lights];
		memset(light_visit[i], 0, num_lights * sizeof(int));
	}

	// allocate cell visit for rl learning
	cell_visit = new int[num_cells];
	memset(cell_visit, 0, num_cells * sizeof(int));

	// allocate proba for q learning
	int num_patches = Q_DIM * Q_DIM;
	cosine_pdf tmp_pdf(vec3(0, 1, 0));
	cuda_rng tmp_rng;
	patch_proba = new float*[num_cells];
	for (int i = 0; i < num_cells; ++i) {
		patch_proba[i] = new float[num_patches];
		for (int j = 0; j < num_patches; ++j) {
			// EXP: cosine or uniform
			//patch_proba[i][j] = BASE_PDF * tmp_pdf.value(tmp_pdf.generate(tmp_rng, rand_state));
			patch_proba[i][j] = BASE_PDF * tmp_rng.generate_random(rand_state);
		}
	}

	// allocate patch visit for q learning
	patch_visit = new int*[num_cells];
	for (int i = 0; i < num_cells; ++i) {
		patch_visit[i] = new int[Q_DIM * Q_DIM];
		memset(patch_visit[i], 0, Q_DIM * Q_DIM * sizeof(int));
	}
#endif

	// add objects into cells
	aabb object_bbox;
	int cell_idx;

	for (int i = 0; i < list_size; ++i) {
		object_bbox = list[i]->get_bounding_box();

		// compute min/max corner cell indices of bounding box
		int ixmin = CLAMP(floorf((object_bbox.pmin.x() - pmin.x()) / (pmax.x() - pmin.x()) * float(nx)), 0, nx - 1);
		int iymin = CLAMP(floorf((object_bbox.pmin.y() - pmin.y()) / (pmax.y() - pmin.y()) * float(ny)), 0, ny - 1);
		int izmin = CLAMP(floorf((object_bbox.pmin.z() - pmin.z()) / (pmax.z() - pmin.z()) * float(nz)), 0, nz - 1);
		int ixmax = CLAMP(floorf((object_bbox.pmax.x() - pmin.x()) / (pmax.x() - pmin.x()) * float(nx)), 0, nx - 1);
		int iymax = CLAMP(floorf((object_bbox.pmax.y() - pmin.y()) / (pmax.y() - pmin.y()) * float(ny)), 0, ny - 1);
		int izmax = CLAMP(floorf((object_bbox.pmax.z() - pmin.z()) / (pmax.z() - pmin.z()) * float(nz)), 0, nz - 1);

		// object state
		printf("add obj%d to cells {", i);

		// assign object to cells
		for (int iz = izmin; iz <= izmax; ++iz) {
			for (int iy = iymin; iy <= iymax; ++iy) {
				for (int ix = ixmin; ix <= ixmax; ++ix) {
					cell_idx = ix + nx * iy + nx * ny * iz;

					// first object
					if (counts[cell_idx] == 0) {
						counts[cell_idx] += 1;
						cells[cell_idx] = list[i];
					}
					else {
						// change to hitable_list
						if (counts[cell_idx] == 1) {
							counts[cell_idx] += 1;
							hitable** object_list = new hitable*[counts[cell_idx]];
							object_list[0] = cells[cell_idx];
							object_list[1] = list[i];

							cells[cell_idx] = new hitable_list(object_list, counts[cell_idx]);
						}
						// append to hitable_list
						else {
							counts[cell_idx] += 1;
							hitable_list* list_ptr = (hitable_list*)cells[cell_idx];
							list_ptr->append(list[i]);
						}
					}
					// cell state
					printf("%d, ", cell_idx);
				}
			}
		}
		printf("}\n");
	}

	// statistics
	int num_zeroes = 0;
	int num_ones = 0;
	int num_twos = 0;
	int num_threes = 0;
	int num_greater = 0;

	for (int i = 0; i < num_cells; ++i) {
		switch (counts[i]) {
		case 0:
			++num_zeroes; break;
		case 1:
			++num_ones; break;
		case 2:
			++num_twos; break;
		case 3:
			++num_threes; break;
		default:
			++num_greater;
		}
	}

	printf("\nnum_zeros: %d\nnum_ones: %d\nnum_twos: %d\nnum_threes: %d\nnum_greater: %d\n",
		num_zeroes, num_ones, num_twos, num_threes, num_greater);

	// free pointer
	delete[] counts;
}

__device__ bool regular_grid::hit(const ray& r, float t_min, float t_max, hit_record& hrec)
{
	float ox = r.o().x();
	float oy = r.o().y();
	float oz = r.o().z();

	float dx = r.d().x();
	float dy = r.d().y();
	float dz = r.d().z();

	x0 = bbox.pmin.x();
	y0 = bbox.pmin.y();
	z0 = bbox.pmin.z();

	x1 = bbox.pmax.x();
	y1 = bbox.pmax.y();
	z1 = bbox.pmax.z();

	// compute min/max t value in x,y,z-coordinates
	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;

	float inv_dx = 1.f / dx;
	tx_min = (dx >= 0.f) ? (x0 - ox) * inv_dx : (x1 - ox) * inv_dx;
	tx_max = (dx >= 0.f) ? (x1 - ox) * inv_dx : (x0 - ox) * inv_dx;

	float inv_dy = 1.f / dy;
	ty_min = (dy >= 0.f) ? (y0 - oy) * inv_dy : (y1 - oy) * inv_dy;
	ty_max = (dy >= 0.f) ? (y1 - oy) * inv_dy : (y0 - oy) * inv_dy;

	float inv_dz = 1.f / dz;
	tz_min = (dz >= 0.f) ? (z0 - oz) * inv_dz : (z1 - oz) * inv_dz;
	tz_max = (dz >= 0.f) ? (z1 - oz) * inv_dz : (z0 - oz) * inv_dz;

	// if the ray misses the grid's bounding box
	float t0 = fmaxf(tx_min, fmaxf(ty_min, tz_min));
	float t1 = fminf(tx_max, fminf(ty_max, tz_max));
	if (t0 > t1) return false;


	// whether the ray starts inside the grid
	int ix, iy, iz;

	if (bbox.inside(r.o())) {
		ix = CLAMP(floorf((ox - x0) / (x1 - x0) * float(nx)), 0, nx - 1);
		iy = CLAMP(floorf((oy - x0) / (y1 - y0) * float(ny)), 0, ny - 1);
		iz = CLAMP(floorf((oz - z0) / (z1 - z0) * float(nz)), 0, nz - 1);
	}
	else {
		vec3 p = r.o() + t0 * r.d();
		ix = CLAMP(floorf((p.x() - x0) / (x1 - x0) * float(nx)), 0, nx - 1);
		iy = CLAMP(floorf((p.y() - y0) / (y1 - y0) * float(ny)), 0, ny - 1);
		iz = CLAMP(floorf((p.z() - z0) / (z1 - z0) * float(nz)), 0, nz - 1);
	}

	// compute step size, terminal condition, and initial next t value
	float dtx = (tx_max - tx_min) / float(nx);
	float dty = (ty_max - ty_min) / float(ny);
	float dtz = (tz_max - tz_min) / float(nz);

	float tx_next, ty_next, tz_next;
	int ix_step, iy_step, iz_step;
	int ix_stop, iy_stop, iz_stop;

	if (dx > 0.f) {
		tx_next = tx_min + float(ix + 1) * dtx;
		ix_step = 1;
		ix_stop = nx;
	}
	else if (dx < 0.f) {
		tx_next = tx_min + float(nx - ix) * dtx;
		ix_step = -1;
		ix_stop = -1;
	}
	else {
		tx_next = CUDART_TWO_TO_126_F;
		ix_step = -1;
		ix_stop = -1;
	}

	if (dy > 0.f) {
		ty_next = ty_min + float(iy + 1) * dty;
		iy_step = 1;
		iy_stop = ny;
	}
	else if (dy < 0.f) {
		ty_next = ty_min + float(ny - iy) * dty;
		iy_step = -1;
		iy_stop = -1;
	}
	else {
		ty_next = CUDART_TWO_TO_126_F;
		iy_step = -1;
		iy_stop = -1;
	}

	if (dz > 0.f) {
		tz_next = tz_min + float(iz + 1) * dtz;
		iz_step = 1;
		iz_stop = nz;
	}
	else if (dz < 0.f) {
		tz_next = tz_min + float(nz - iz) * dtz;
		iz_step = -1;
		iz_stop = -1;
	}
	else {
		tz_next = CUDART_TWO_TO_126_F;
		iz_step = -1;
		iz_stop = -1;
	}


	// traverse the grid (3DDDA)
	while (true) {
		hitable* object_ptr = cells[ix + nx * iy + nx * ny * iz];

		// x-face
		if (tx_next < ty_next && tx_next < tz_next) {
			if (object_ptr && object_ptr->hit(r, t_min, t_max, hrec) && hrec.t < tx_next) {
				return true;
			}
			tx_next += dtx;
			ix += ix_step;

			if (ix == ix_stop) {
				return false;
			}
		}
		else {
			// y-face
			if (ty_next < tz_next) {
				if (object_ptr && object_ptr->hit(r, t_min, t_max, hrec) && hrec.t < ty_next) {
					return true;
				}
				ty_next += dty;
				iy += iy_step;

				if (iy == iy_stop) {
					return false;
				}
			}
			// z-face
			else {
				if (object_ptr && object_ptr->hit(r, t_min, t_max, hrec) && hrec.t < tz_next) {
					return true;
				}
				tz_next += dtz;
				iz += iz_step;

				if (iz == iz_stop) {
					return false;
				}
			}
		}
	}
}

__device__ inline void regular_grid::dump_proba()
{
	FILE* proba_fp;
	errno_t proba_err = fopen_s(&proba_fp, "proba.txt", "w");

	if (proba_err != 0) {
		printf("Failed to create file.\n");
	}
	else {
		float* cdf = new float[Q_DIM * Q_DIM + 1];

		// ENTRY
		fprintf(proba_fp, "num_cell: %d, num_lights: %d\n", nx * ny * nz, light_size);
		fprintf(proba_fp, "cell id ");
		for (int j = 0; j < Q_DIM * Q_DIM; ++j) {
			fprintf(proba_fp, " pr%-5d ", j);
		}
		for (int j = 0; j < Q_DIM * Q_DIM + 1; ++j) {
			fprintf(proba_fp, " pr%-5d ", j);
		}
		fprintf(proba_fp, "\n");

		for (int i = 0; i < nx * ny * nz; ++i) {
			fprintf(proba_fp, "%7d: ", i);

			// PDF
			for (int j = 0; j < Q_DIM * Q_DIM; ++j) {
				fprintf(proba_fp, "%7.6f ", patch_proba[i][j]);
			}

			// CDF
			cdf[0] = 0.f;
			for (int j = 1; j < Q_DIM * Q_DIM + 1; ++j) {
				cdf[j] = patch_proba[i][j - 1] + cdf[j - 1];
			}
			for (int j = 0; j < Q_DIM * Q_DIM + 1; ++j) {
				cdf[j] /= cdf[Q_DIM * Q_DIM];
				fprintf(proba_fp, "%7.6f ", cdf[j]);
			}

			fprintf(proba_fp, "\n");
		}

		delete[] cdf;
		fclose(proba_fp);
	}
}

