/* ==========================================================================
   cudaHelloWorld.cu
   ==========================================================================

   Test hello world to make sure that CUDA is working

*/

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "textureChanger.h"
#include "utils.h"


// --------------------------------------------------------------------------
// Kernel
// --------------------------------------------------------------------------

__global__ void CudaKernelHelloWorld(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}


// --------------------------------------------------------------------------
// Wrapper
// --------------------------------------------------------------------------

extern "C"
void CudaWrapperHelloWorld()
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
	CudaKernelHelloWorld << <dimGrid, dimBlock >> >(ad, bd);
	cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);
	cudaFree(ad);

	output += a;

	DebugInUnity(output);
}