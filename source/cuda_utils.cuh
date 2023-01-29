#pragma once

#include "cuda_helper.cuh"
#include "hitable_list.cuh"
#include "material.cuh"
#include "aarect.cuh"
#include "aabox.cuh"
#include "sphere.cuh"
#include "tri.cuh"
#include "regular_grid.cuh"
#include "camera.cuh"
#include "path_tracer.cuh"
#include "rng.cuh"

__device__ inline vec3 de_nan(const vec3& val)
{
	vec3 tmp = val;
	if (!(tmp[0] == tmp[0])) tmp[0] = 0;
	if (!(tmp[1] == tmp[1])) tmp[1] = 0;
	if (!(tmp[2] == tmp[2])) tmp[2] = 0;
	return tmp;
}

__device__ void cornell_box(hitable_list** dev_scene, hitable_list** dev_light, camera** dev_cam, path_tracer** dev_pt,
							rng** dev_rng, curandState* dev_rand_states)
{
	// OBJECT
	int i = 0;
	hitable** scene_list = new hitable*[9];

	material* red = new lambertian(new constant_texture(vec3(0.65f, 0.05f, 0.05f)));
	material* white = new lambertian(new constant_texture(vec3(0.73f, 0.73f, 0.73f)));
	material* green = new lambertian(new constant_texture(vec3(0.12f, 0.45f, 0.15f)));
	material* blue = new lambertian(new constant_texture(vec3(0.1f, 0.2f, 0.5f)));
	material* gold = new lambertian(new constant_texture(vec3(0.831f, 0.686f, 0.216f)));
	material* light = new area_light(new constant_texture(vec3(5.f)));

	scene_list[i++] = new flip_normal(new yz_rect(0, 555, 0, 555, 555, green));
	scene_list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
	scene_list[i++] = new flip_normal(new xz_rect(0, 555, 0, 555, 555, white));
	scene_list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
	scene_list[i++] = new flip_normal(new xy_rect(0, 555, 0, 555, 555, white));
	scene_list[i++] = new sphere(vec3(190, 90, 190), 90, blue);
	scene_list[i++] = new tri(vec3(300, 40, 80), vec3(450, 40, 80), vec3(375, 135, 155), gold);
	scene_list[i++] = new aabox(vec3(265, 0, 295), vec3(430, 330, 460), white);

	// LIGHT
	int j = 0;
	hitable** light_list = new hitable*[1];

	scene_list[i++] = new flip_normal(new xz_rect(213, 343, 227, 332, 554, light, j));
	light_list[j++] = new xz_rect(213, 343, 227, 332, 554, nullptr);

	*dev_light = new hitable_list(light_list, j);

	// regular grid
	regular_grid* grid = new regular_grid(scene_list, i);
	grid->setup_cells(&dev_rand_states[0], j);

	*dev_scene = grid;

	// camera
	vec3 lookfrom(278, 278, -800);
	vec3 lookat(278, 278, 0);
	float vfov = 40.f;
	*dev_cam = new camera(lookfrom, lookat, vec3(0.f, 1.f, 0.f), vfov, (float)WIDTH / (float)HEIGHT);

	// path tracer
	*dev_pt = new path_tracer();

	// random number generator
	*dev_rng = new cuda_rng();
}

__device__ inline void indirect_scene(hitable_list** dev_scene, hitable_list** dev_light, camera** dev_cam, path_tracer** dev_pt,
									  rng** dev_rng, curandState* dev_rand_states)
{
	// OBJECT
	int i = 0;
	hitable** scene_list = new hitable*[13];

	material* red = new lambertian(new constant_texture(vec3(0.65f, 0.05f, 0.05f)));
	material* white = new lambertian(new constant_texture(vec3(0.73f, 0.73f, 0.73f)));
	material* green = new lambertian(new constant_texture(vec3(0.12f, 0.45f, 0.15f)));
	material* blue = new lambertian(new constant_texture(vec3(0.1f, 0.2f, 0.5f)));
	material* light = new area_light(new constant_texture(vec3(5.f)));

	// -x
	scene_list[i++] = new flip_normal(new yz_rect(0, 600, 0, 800, 600, green));
	scene_list[i++] = new flip_normal(new yz_rect(0, 600, 500, 800, 270, white));
	// +x
	scene_list[i++] = new yz_rect(0, 600, 0, 800, 0, red);
	scene_list[i++] = new yz_rect(0, 600, 500, 800, 330, white);
	// +z
	scene_list[i++] = new xy_rect(0, 600, 0, 600, 0, white);
	// -z
	scene_list[i++] = new flip_normal(new xy_rect(330, 600, 0, 600, 800, white));
	scene_list[i++] = new flip_normal(new xy_rect(270, 330, 0, 600, 500, white));
	scene_list[i++] = new flip_normal(new xy_rect(0, 30, 0, 600, 800, white));
	scene_list[i++] = new flip_normal(new xy_rect(240, 270, 0, 600, 800, white));
	// +y
	scene_list[i++] = new xz_rect(0, 600, 0, 800, 0, white);
	// -y
	scene_list[i++] = new flip_normal(new xz_rect(0, 600, 0, 800, 600, white));

	// other
	scene_list[i++] = new sphere(vec3(465, 90, 650), 90, blue);

	// LIGHT
	int j = 0;
	hitable** light_list = new hitable*[1];

	scene_list[i++] = new flip_normal(new xy_rect(0, 270, 0, 600, 820, light, j));
	light_list[j++] = new xy_rect(0, 270, 0, 600, 820, light);

	*dev_light = new hitable_list(light_list, j);

	// regular grid
	regular_grid* grid = new regular_grid(scene_list, i);
	grid->setup_cells(&dev_rand_states[0], j);

	*dev_scene = grid;

	// camera
	vec3 lookfrom(300, 300, 5);
	vec3 lookat(300, 300, 300);
	float vfov = 75.f;
	*dev_cam = new camera(lookfrom, lookat, vec3(0.f, 1.f, 0.f), vfov, (float)WIDTH / (float)HEIGHT);

	// path tracer
	*dev_pt = new path_tracer();

	// random number generator
	*dev_rng = new cuda_rng();
}

__device__ inline void direct_scene(hitable_list** dev_scene, hitable_list** dev_light, camera** dev_cam, path_tracer** dev_pt,
							 rng** dev_rng, curandState* dev_rand_states)
{
	// OBJECT
	int i = 0;
	hitable** scene_list = new hitable*[81];

	material* red = new lambertian(new constant_texture(vec3(0.65f, 0.05f, 0.05f)));
	material* white = new lambertian(new constant_texture(vec3(0.73f, 0.73f, 0.73f)));
	material* green = new lambertian(new constant_texture(vec3(0.12f, 0.45f, 0.15f)));
	material* light = new area_light(new constant_texture(vec3(7.5f)));

	// box
	// +x
	for (int x = 0; x < 8; ++x) {
		// +z
		for (int z = 0; z < 6; ++z) {
			scene_list[i++] = new aabox(vec3(100 + x * 200, 0, 50 + z * 350), vec3(200 + x * 200, 100, 150 + z * 350), (x + z) % 2 ? green : red);
		}
	}

	// plane
	scene_list[i++] = new xz_rect(0, 1800, 0, 1800, 0, white);

	// LIGHT
	int j = 0;
	hitable** light_list = new hitable*[32];

	// +x
	for (int x = 0; x < 8; ++x) {
		// +z
		for (int z = 0; z < 4; ++z) {
			scene_list[i++] = new sphere(vec3(150 + x * 200, 50, 275 + z * 350), 25, light, j);
			light_list[j++] = new sphere(vec3(150 + x * 200, 50, 275 + z * 350), 25, nullptr);
		}
	}
	*dev_light = new hitable_list(light_list, j);

	// regular grid
	regular_grid* grid = new regular_grid(scene_list, i);
	grid->setup_cells(&dev_rand_states[0], j);

	*dev_scene = grid;

	// camera
	vec3 lookfrom(900, 800, 900);
	vec3 lookat(900, 0, 900);
	float vfov = 75.f;
	*dev_cam = new camera(lookfrom, lookat, vec3(0.f, 1.f, 0.f), vfov, (float)WIDTH / (float)HEIGHT);

	// path tracer
	*dev_pt = new path_tracer();

	// random number generator
	*dev_rng = new cuda_rng();
}

