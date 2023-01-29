#pragma once

#include "vec3.cuh"

struct interaction
{
	const float MOUSE_SPEED = 0.03f;
	const float KEYBOARD_SPEED = 5.f;
	const float DEG2RAD = 0.0174533f;
	const vec3 UP = vec3(0.f, 1.f, 0.f);

	bool is_mouse = false;
	bool is_keyboard = false;
	int curr_x, curr_y;
	float curr_yaw = 90.f, curr_pitch = 0.f;
	vec3 lookdir = vec3(0.f, 0.f, 1.f);
	vec3 lookfrom = vec3(0.f);
};

