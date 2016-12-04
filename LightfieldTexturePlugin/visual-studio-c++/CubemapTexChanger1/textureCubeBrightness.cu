/* ==========================================================================
   textureCubeBrightness.cu
   ==========================================================================

   Main wrapper + kernel that changes the brightness of each face

*/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"


#define PI 3.1415926536f

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
template <typename T>
__device__ T clip(const T& n, const T& lower, const T& upper) {
	return max(lower, min(n, upper));
}


// --------------------------------------------------------------------------
// Kernel
// --------------------------------------------------------------------------

// Varies brightness of output pixel based on parameters
__global__ void CudaKernelTextureCubeBrightness(char *inputSurface, char *outputSurface, int width, int height, size_t pitch, int face, float t, float brightnessIntervalPeriod)
{
	unsigned char *inputPixel;
	unsigned char *outputPixel;

	// Grid indices
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// In the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// Get a pointer to this input / output pixel
	inputPixel = (unsigned char *)(inputSurface + y*pitch) + 4 * x;
	outputPixel = (unsigned char *)(outputSurface + y*pitch) + 4 * x;

	// Set brightness multiplier
	float brightnessMultipierA = 3;
	float brightnessMultiplierK = 2 * PI / brightnessIntervalPeriod;

	float brightnessMultiplier = 1 + brightnessMultipierA * sin(brightnessMultiplierK * t);

	// Set ouput pixel
	outputPixel[0] = clip((int)(inputPixel[0] * brightnessMultiplier), 0, 255);
	outputPixel[1] = clip((int)(inputPixel[1] * brightnessMultiplier), 0, 255);
	outputPixel[2] = clip((int)(inputPixel[2] * brightnessMultiplier), 0, 255);
	outputPixel[3] = inputPixel[3];

	// Debug - set equal to input
	//outputPixel[0] = inputPixel[0];
	//outputPixel[1] = inputPixel[1];
	//outputPixel[2] = inputPixel[2];
	//outputPixel[3] = inputPixel[3];

}


// --------------------------------------------------------------------------
// Wrapper
// --------------------------------------------------------------------------

// Sets up grid / blocks, launches kernel
extern "C"
void CudaWrapperTextureCubeBrightness(void *inputSurface, void *outputSurface, int width, int height, size_t pitch, int face, float t, float brightnessIntervalPeriod)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	CudaKernelTextureCubeBrightness <<<Dg, Db >>>((char *)inputSurface, (char *)outputSurface, 
												    width, height, pitch, face, t, brightnessIntervalPeriod);
	ProcessCudaError("cuda_kernel_texture_cube() failed to launch error: ");
}


