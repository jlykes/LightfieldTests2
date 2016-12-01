#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define PI 3.1415926536f

// --------------------------------------------------------------------------
// Main Kernel / wrapper


/*
* Paint a 2D surface with a moving bulls-eye pattern.  The "face" parameter selects
* between 6 different colors to use.  We will use a different color on each face of a
* cube map.
*/
__global__ void cuda_kernel_texture_cube(char *surface, int width, int height, size_t pitch, int face, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned char *pixel;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to this pixel
	pixel = (unsigned char *)(surface + y*pitch) + 4 * x;

	// populate it
	float theta_x = (2.0f*x) / width - 1.0f;
	float theta_y = (2.0f*y) / height - 1.0f;
	float theta = 2.0f*PI*sqrt(theta_x*theta_x + theta_y*theta_y);
	unsigned char value = 255 * (0.6f + 0.4f*cos(theta + t));

	pixel[3] = 255; // alpha

	if (face % 2)
	{
		pixel[0] =    // blue
			pixel[1] =    // green
			pixel[2] = 0.5; // red
		pixel[face / 2] = value;
	}
	else
	{
		pixel[0] =        // blue
			pixel[1] =        // green
			pixel[2] = value; // red
		pixel[face / 2] = 0.5;
	}
}


extern "C"
void cuda_texture_cube(void *surface, int width, int height, size_t pitch, int face, float t)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	cuda_kernel_texture_cube <<<Dg, Db >>>((char *)surface, width, height, pitch, face, t);
	ProcessCudaError("cuda_kernel_texture_cube() failed to launch error: ");
}


// --------------------------------------------------------------------------
// CUDA testing - Hello World


__global__ void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}

extern "C"
void cuda_test()
{
	DebugInUnity("This is testing to see if I can call code from texture_cube.cu");

	const int N = 7;
	const int blocksize = 7;

	char a[N] = "Hello ";
	int b[N] = { 15, 10, 6, 0, -11, 1, 0 };

	char *ad;
	int *bd;
	const int csize = N * sizeof(char);
	const int isize = N * sizeof(int);

	std::string output;
	output += a;

	cudaMalloc((void**)&ad, csize);
	cudaMalloc((void**)&bd, isize);
	cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, isize, cudaMemcpyHostToDevice);

	dim3 dimBlock(blocksize, 1);
	dim3 dimGrid(1, 1);
	hello << <dimGrid, dimBlock >> >(ad, bd);
	cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);
	cudaFree(ad);

	output += a;

	DebugInUnity(output);
}