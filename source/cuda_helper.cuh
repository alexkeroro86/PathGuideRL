#pragma once

// opengl
#include <GL/glew.h>
#include <GL/freeglut.h>

// c
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

// c++
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

// cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cuda_gl_interop.h> // after included opengl

// opengl
#pragma comment (lib, "glew32.lib")
#pragma comment (lib, "freeglut.lib")

// cuda
#pragma comment (lib, "cublas.lib")
#pragma comment (lib, "cuda.lib")
#pragma comment (lib, "cudadevrt.lib")
#pragma comment (lib, "cudart.lib")
#pragma comment (lib, "cudart_static.lib")
//#pragma comment (lib, "nvcuvid.lib")
#pragma comment (lib, "OpenCL.lib")

inline void e(
	cudaError_t err,
	const char* file,
	int line
)
{
	if (err != cudaSuccess) {
		printf("Error in %s at line %d:\n\t%s\n", file, line, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR(err) ( e(err, __FILE__, __LINE__) )

inline void print_gpu_info()
{
	cudaDeviceProp prop;
	int count;

	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; ++i) {
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

		printf("=== General Information for device %d ===\n", i);
		printf("Name:  %s\n", prop.name);
		printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
		printf("Clock rate:  %d\n", prop.clockRate);
		printf("Device copy overlap:  %s\n", prop.deviceOverlap ? "Enabled" : "Disabled");
		printf("Kernel execition timeout:  %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

		printf("=== Memory Information for device %d ===\n", i);
		printf("Total global mem:  %lld\n", prop.totalGlobalMem);
		printf("Total constant mem:  %ld\n", prop.totalConstMem);
		printf("Max mem pitch:  %ld\n", prop.memPitch);
		printf("Texture alignment:  %ld\n", prop.textureAlignment);

		printf("=== MP Information for device %d ===\n", i);
		printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp:  %d\n", prop.regsPerBlock);
		printf("Threads in wrap:  %d\n", prop.warpSize);
		printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

		printf("\n");
	}
}

