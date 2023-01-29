#pragma once

#include "vec3.cuh"

#define CLAMP(v, lo, hi) (((v) < (lo)) ? (lo) : ((hi) < (v)) ? (hi) : (v))

#define PI 3.1415926f
// NOTE: IF SOMETHING DISAPPEARS, THEN DONT USE TOO SMALL
#define EPSILON 0.0001f

#define WIDTH 640
#define HEIGHT 480
#define DEPTH 16
#define MAX_SAMPLES 256
#define TILE 2

// FLAG: use rl-based rendering or not
#define USE_RL 1
// FLAG: print internal log
#define DEBUG 0
// FLAG: test scene
#define USE_INDIRECT_SCENE
//#define USE_DIRECT_SCENE
//#define USE_CORNELL_BOX
// HYPERPARAMETER: learning rate for updating pdf (update weights) / td, q
#define ETA 0.85f
// HYPERPARAMETER: trade-off between exploration and exploitation for sampling (rl-related parameter) / td, q
#define BASE_PDF 0.15f
// HYPERPARAMETER: grid resolution (model complexity) / td, q
#define GRID_MULTIPLIER 4.f

// HYPERPARAMETER: discretization of square resolution for mapping to hemisphere / q
#define Q_DIM 3

