/* ==========================================================================
   utils.cpp
   ==========================================================================

   Implementation of basic utils

*/

#include "utils.h"

// --------------------------------------------------------------------------
// Setup for making Debug.Log in Unity callable from here
// --------------------------------------------------------------------------

// Definitions
typedef void(__stdcall * DebugCallback) (const char *);
DebugCallback gDebugCallback;
void DebugInUnity(std::string message);

// Links to Unity function of same name, to set Unity delegate to C++ plugin fn pointer
extern "C" void __declspec(dllexport) RegisterDebugCallback(DebugCallback callback)
{
	gDebugCallback = callback;
}

// Sends string "message" to Unity to print via Debug.Log
void DebugInUnity(std::string message)
{
	gDebugCallback(message.c_str());
}


// --------------------------------------------------------------------------
// Other helpers
// --------------------------------------------------------------------------

// Takes last CUDA error, sends to Unity Debug.Log for viewing
void ProcessCudaError(std::string prefix)
{
	cudaError_t error = cudaSuccess;
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		DebugInUnity(prefix + std::to_string(error));
	}
}

// Prints the parameters of a texture to Unity console
void PrintTextureDesc(D3D11_TEXTURE2D_DESC desc)
{
	DebugInUnity("Width: " + std::to_string(desc.Width));
	DebugInUnity("Height: " + std::to_string(desc.Height));
	DebugInUnity("Mip Levels: " + std::to_string(desc.MipLevels));
	DebugInUnity("Array size: " + std::to_string(desc.ArraySize));
	DebugInUnity("Format: " + std::to_string(desc.Format));
	DebugInUnity("SampleDesc.Count: " + std::to_string(desc.SampleDesc.Count));
	DebugInUnity("Usage: " + std::to_string(desc.Usage));
	DebugInUnity("Bind Flags: " + std::to_string(desc.BindFlags));
	DebugInUnity("Misc Flags: " + std::to_string(desc.MiscFlags));
}