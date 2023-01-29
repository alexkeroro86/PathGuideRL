#include "cuda_utils.cuh"
#include "helper.h"

hitable_list** dev_scene;
hitable_list** dev_light;
camera** dev_cam;
path_tracer** dev_pt;
rng** dev_rng;
curandState* dev_rand_states;
float* dev_buf;

__global__ void init_curand(curandState* dev_rand_states)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= WIDTH || y >= HEIGHT) {
		return;
	}
	int idx = x + WIDTH * y;

	curand_init((unsigned long long)clock(), idx, 0, &dev_rand_states[idx]);
}

__global__ void init_scene(hitable_list** dev_scene, hitable_list** dev_light, camera** dev_cam, path_tracer** dev_pt,
					       rng** dev_rng, curandState* dev_rand_states)
{
	if (blockIdx.x == 0 && threadIdx.x == 0) {
#ifdef USE_INDIRECT_SCENE
		indirect_scene(dev_scene, dev_light, dev_cam, dev_pt, dev_rng, dev_rand_states);
#elif defined(USE_DIRECT_SCENE)
		direct_scene(dev_scene, dev_light, dev_cam, dev_pt, dev_rng, dev_rand_states);
#elif defined(USE_CORNELL_BOX)
		cornell_box(dev_scene, dev_light, dev_cam, dev_pt, dev_rng, dev_rand_states);
#elif
#error Please choose an available scene
#endif
	}
}

__global__ void render(float iter, int total, float* dev_tex, hitable_list** dev_scene, hitable_list** dev_light, camera** dev_cam,
					   interaction param, path_tracer** dev_pt, rng** dev_rng, float* dev_buf, curandState* dev_rand_states)
{
	__shared__ curandState rand_states[TILE][TILE];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if (x >= WIDTH || y >= HEIGHT) {
		return;
	}
	int idx = x + WIDTH * y;

	rand_states[tx][ty] = dev_rand_states[(idx * (int)iter) % total];

	(*dev_cam)->update(param.DEG2RAD, param.lookdir, param.lookfrom, param.UP);

	__syncthreads();
	
	regular_grid* rg_ptr = (regular_grid*)(*dev_scene);

	float* norm_patch_pdf = new float[Q_DIM * Q_DIM];
	float* norm_light_pdf = new float[rg_ptr->light_size];
	float r1, r2;

	(*dev_rng)->random_square_point(&rand_states[tx][ty], r1, r2);
	float u = float(x + r1) / float(WIDTH);
	float v = float(y + r2) / float(HEIGHT);
	ray r = (*dev_cam)->get_primary_ray(u, v);
	vec3 color = de_nan((*dev_pt)->trace_ray(r, *dev_scene, *dev_light, norm_patch_pdf, norm_light_pdf, DEPTH, **dev_rng, &rand_states[tx][ty]));

	dev_buf[3 * idx + 0] += color[0];
	dev_buf[3 * idx + 1] += color[1];
	dev_buf[3 * idx + 2] += color[2];

	dev_tex[4 * idx + 0] = sqrtf(dev_buf[3 * idx + 0] / iter);
	dev_tex[4 * idx + 1] = sqrtf(dev_buf[3 * idx + 1] / iter);
	dev_tex[4 * idx + 2] = sqrtf(dev_buf[3 * idx + 2] / iter);
	dev_tex[4 * idx + 3] = 0.f;

	delete[] norm_patch_pdf;
	delete[] norm_light_pdf;
}

extern "C"
void launch_init_curand(dim3 grid, dim3 block)
{
	HANDLE_ERROR(cudaMalloc((void**)&dev_rand_states, WIDTH * HEIGHT * sizeof(curandState)));
	init_curand<<<grid, block>>>(dev_rand_states);
}

extern "C"
void launch_init_scene()
{
	HANDLE_ERROR(cudaMalloc((void**)&dev_scene, sizeof(hitable_list*)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_light, sizeof(hitable_list*)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_cam, sizeof(camera*)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_pt, sizeof(path_tracer*)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_rng, sizeof(rng*)));

	HANDLE_ERROR(cudaMalloc((void**)&dev_buf, WIDTH * HEIGHT * 3 * sizeof(float)));
	HANDLE_ERROR(cudaMemset(dev_buf, 0, WIDTH * HEIGHT * 3 * sizeof(float)));

	init_scene<<<1, 1>>>(dev_scene, dev_light, dev_cam, dev_pt, dev_rng, dev_rand_states);
}

extern "C"
void launch_clear_buf()
{
	HANDLE_ERROR(cudaMemset(dev_buf, 0, WIDTH * HEIGHT * 3 * sizeof(float)));
}

extern "C"
void launch_render(dim3 grid, dim3 block, float iter, float* dev_tex, interaction param)
{
	render<<<grid, block>>>(iter, WIDTH * HEIGHT, dev_tex, dev_scene, dev_light, dev_cam, param, dev_pt, dev_rng, dev_buf, dev_rand_states);
}

extern "C"
void launch_free_device()
{
	HANDLE_ERROR(cudaFree(dev_rand_states));

	HANDLE_ERROR(cudaFree(dev_scene));
	HANDLE_ERROR(cudaFree(dev_light));
	HANDLE_ERROR(cudaFree(dev_cam));
	HANDLE_ERROR(cudaFree(dev_pt));
	HANDLE_ERROR(cudaFree(dev_rng));

	HANDLE_ERROR(cudaFree(dev_buf));
}

