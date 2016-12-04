/* ==========================================================================
   unityInterfaceFunctions.cpp
   ==========================================================================
   
   MAIN file - holds definition/implementation for all functions that link
   directly to Unity

*/


// --------------------------------------------------------------------------
// General include headers
// --------------------------------------------------------------------------

#include "Unity/IUnityGraphics.h"
#include "unityInterfaceFunctions.h"
#include "utils.h"
#include "unityPluginSetup.h"
#include "textureChanger.h"
#include "DDSTextureLoader.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>


// --------------------------------------------------------------------------
// Include headers for the graphics APIs we support
// --------------------------------------------------------------------------

#if SUPPORT_D3D11
#	include <d3d11.h>
#	include "Unity/IUnityGraphicsD3D11.h"
#endif


// --------------------------------------------------------------------------
// Include stuff for CUDA
// --------------------------------------------------------------------------

#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h


// -------------------------------------------------------------------------------------------------------
// =======================================================================================================
// UNITY INTERFACE FUNCTIONS
// =======================================================================================================
// -------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------
// UnitySetInterfaces
// --------------------------------------------------------------------------

extern "C" void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
	s_UnityInterfaces = unityInterfaces;
	s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();
	s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

	// Run OnGraphicsDeviceEvent(initialize) manually on plugin load
	OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
	s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}


// --------------------------------------------------------------------------
// GraphicsDeviceEvent
// --------------------------------------------------------------------------

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
	//DebugInUnity("OnGraphicsDeviceEvent called");
	UnityGfxRenderer currentDeviceType = s_DeviceType;

	switch (eventType)
	{
	case kUnityGfxDeviceEventInitialize:
	{
		//DebugInUnity("OnGraphicsDeviceEvent(Initialize).\n");

		// This seems to be returning what type of rendering device we are using (OpenGL, D3D 11, XBoxOne, etc.)
		s_DeviceType = s_Graphics->GetRenderer();
		currentDeviceType = s_DeviceType;
		break;
	}

	case kUnityGfxDeviceEventShutdown:
	{
		//DebugInUnity("OnGraphicsDeviceEvent(Shutdown).\n");
		s_DeviceType = kUnityGfxRendererNull;
		break;
	}

	case kUnityGfxDeviceEventBeforeReset:
	{
		//DebugInUnity("OnGraphicsDeviceEvent(BeforeReset).\n");
		break;
	}

	case kUnityGfxDeviceEventAfterReset:
	{
		//DebugInUnity("OnGraphicsDeviceEvent(AfterReset).\n");
		break;
	}
	};

	//This seems to be the only reason we care about all of the graphics device stuff - 
	//so we can tell whether running DirectX, etc. and use that to define what our code
	//should look like
	if (currentDeviceType == kUnityGfxRendererD3D11)
		DoEventGraphicsDeviceD3D11(eventType);
}


// --------------------------------------------------------------------------
//  Direct3D 11 setup/teardown code
// --------------------------------------------------------------------------

static void DoEventGraphicsDeviceD3D11(UnityGfxDeviceEventType eventType)
{
	if (eventType == kUnityGfxDeviceEventInitialize)
	{
		IUnityGraphicsD3D11* d3d11 = s_UnityInterfaces->Get<IUnityGraphicsD3D11>();
		g_D3D11Device = d3d11->GetDevice();
	}
}

// --------------------------------------------------------------------------
// Set time / streaming assets path
// --------------------------------------------------------------------------


extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTimeFromUnity(float t)
{
	g_Time = t;
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetUnityStreamingAssetsPath(const char* path)
{
	s_UnityStreamingAssetsPath = path;
}


// --------------------------------------------------------------------------
// Rendering / texture setting
// --------------------------------------------------------------------------

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
	return OnRenderEvent;
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* texturePtr, const char* eyeName)
{
	SetTextureFromUnityImplementation(texturePtr, eyeName);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetBrightnessIntervalPeriodFromUnity(float brightnessIntervalPeriod)
{
	g_brightnessPeriodInterval = brightnessIntervalPeriod;
}

// This will be called for GL.IssuePluginEvent script calls; eventID will
// be the integer passed to IssuePluginEvent. In this example, we just ignore
// that value.
static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	// Unknown graphics device type? Do nothing.
	if (s_DeviceType == kUnityGfxRendererNull)
		return;

	if (eventID == 1)
	{
		DoRendering(kLeftEyeName);
	}
	else if (eventID == 2)
	{
		DoRendering(kRightEyeName);
	}
}


// --------------------------------------------------------------------------
// Console logging
// --------------------------------------------------------------------------

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API LogConsoleOutputToConsole()
{
	FILE * pConsole;
	AllocConsole();
	freopen_s(&pConsole, "CONOUT$", "wb", stdout);
	freopen_s(&pConsole, "CONOUT$", "wb", stderr);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API LogConsoleOutputToFile()
{
	FILE * pConsole;
	AllocConsole();
	//freopen_s(&pConsole, "CONOUT$", "wb", stdout);
	//freopen_s(&pConsole, "CONOUT$", "wb", stderr);
	freopen("out.txt", "w", stdout);
	freopen("out.txt", "w", stderr);
}

// -------------------------------------------------------------------------------------------------------
// =======================================================================================================
// TEXTURE / RENDER HANDLING
// =======================================================================================================
// -------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Texture setup / filling
// --------------------------------------------------------------------------

// Implementation of texture setting function that Unity calls
static void SetTextureFromUnityImplementation(void* texturePtr, std::string eyeName) 
{
	struct TextureCube * g_texture_cube = GetTextureCubeForEye(eyeName);
	LoadTextureIntoPointers(texturePtr, g_texture_cube, eyeName);

	// Puts pTexture into cudaResource
	cudaGraphicsD3D11RegisterResource(&g_texture_cube->cudaResourceOutput, g_texture_cube->pTexture, cudaGraphicsRegisterFlagsNone);
	ProcessCudaError("cudaGraphicsD3D11RegisterResource (g_texture_cube) failed for eye [" + eyeName + "]: ");

	//--- Output buffers ---

	// Create the output buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
	cudaMallocPitch(&g_texture_cube->cudaLinearMemoryOutput, &g_texture_cube->pitch, g_texture_cube->size * 4, g_texture_cube->size);
	ProcessCudaError("cudaMallocPitch (g_texture_cube output) failed for eye [" + eyeName + "]: ");
	cudaMemset(g_texture_cube->cudaLinearMemoryOutput, 128, g_texture_cube->pitch * g_texture_cube->size);
	ProcessCudaError("cudaMemset (g_texture_cube output) failed for eye [" + eyeName + "]: ");

	//---- Input buffers ---

	// Map CUDA resource
	cudaStream_t    stream = 0;
	const int nbResources = 1;
	cudaGraphicsResource *ppResources[nbResources] =
	{
		g_texture_cube->cudaResourceOutput
	};
	cudaGraphicsMapResources(nbResources, ppResources, stream);
	ProcessCudaError("cudaGraphicsMapResources failed for [" + eyeName + "]:");

	// Allocate input texture
	AllocateInputCubemapFaces(g_texture_cube, eyeName);

	// Unmap the resources
	cudaGraphicsUnmapResources(nbResources, ppResources, stream);
	ProcessCudaError("cudaGraphicsUnmapResources failed for [" + eyeName + "]:");
}


// Run kernels in CUDA file
void RunTextureFillingKernels(std::string eyeName)
{
	struct TextureCube * g_texture_cube = GetTextureCubeForEye(eyeName);
	printf("This is a test print to see if I can see console. Kernel running for eyeName %s \n", eyeName);

	static float t = 0.0f;
	for (int face = 0; face < 6; ++face)
	{
		// Set up cuArray because this is the way you can translate the cudaLinearMemory (which CUDA
		// can operate on) to the cudaArray/cudaResource format (which CUDA can't operate on, apparently)
		cudaArray *cuArrayForOutput;
		cudaGraphicsSubResourceGetMappedArray(&cuArrayForOutput, g_texture_cube->cudaResourceOutput, face, 0);
		ProcessCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_cube) failed: ");

		// Kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
		CudaWrapperTextureCubeBrightness(g_texture_cube->cudaLinearMemoryInput[face], 
										 g_texture_cube->cudaLinearMemoryOutput, 
										 g_texture_cube->size, g_texture_cube->size, 
										 g_texture_cube->pitch, face, g_Time, g_brightnessPeriodInterval);
		ProcessCudaError("cuda_texture_cube failed: ");

		// Then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
		cudaMemcpy2DToArray(
			cuArrayForOutput, // dst array
			0, 0,    // offset
			g_texture_cube->cudaLinearMemoryOutput, g_texture_cube->pitch, // src
			g_texture_cube->size * 4, g_texture_cube->size,            // extent
			cudaMemcpyDeviceToDevice); // kind
		ProcessCudaError("cudaMemcpy2DToArray failed: ");
	}
	t += 0.1f;
}


// Loads texture from dds file, assigns g_texture_cube pointer to left or right eye
void LoadTextureIntoPointers(void* texturePtr, struct TextureCube * g_texture_cube, std::string eyeName) 
{
	// Initialize g_texture_cube to by Unity texture; set other parameters
	g_texture_cube->pTexture = (ID3D11Texture2D*)texturePtr;
	D3D11_TEXTURE2D_DESC desc;
	g_texture_cube->pTexture->GetDesc(&desc);
	g_texture_cube->size = desc.Width;

	// Load texture from file using DDSTextureLoader
	D3DX11_IMAGE_LOAD_INFO loadedTextureLoadInfo;
	loadedTextureLoadInfo.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;
	loadedTextureLoadInfo.MipLevels = 1;

	ID3D11Texture2D* loadedTexture = 0;
	std::string loadedTextureFilename = (eyeName == kLeftEyeName) ? "L-walk_to_me.dds" : "R-walk_to_me.dds"; //Assumes in root folder of Unity project

	HRESULT hr = D3DX11CreateTextureFromFile(g_D3D11Device, loadedTextureFilename.c_str(), &loadedTextureLoadInfo, 0, (ID3D11Resource**)&loadedTexture, 0);

	//Copy over to current texture pointer
	ID3D11DeviceContext* ctx = NULL;
	g_D3D11Device->GetImmediateContext(&ctx);
	ctx->CopyResource(g_texture_cube->pTexture, loadedTexture);
	ctx->Release();
}

// For each face, allocates linear memory space on GPU, and copies pixels from original texture to allocated space
void AllocateInputCubemapFaces(struct TextureCube * g_texture_cube, std::string eyeName)
{
	// Create the input buffers
	for (int face = 0; face < 6; ++face)
	{
		// Malloc space for face
		cudaMallocPitch(&g_texture_cube->cudaLinearMemoryInput[face], &g_texture_cube->pitch, g_texture_cube->size * 4, g_texture_cube->size);
		ProcessCudaError("cudaMallocPitch (g_texture_cube input) failed for eye [" + eyeName + "], face [" + std::to_string(face) + "]: ");

		// Link original texture with cuArray so can access pixels via cuArray
		cudaArray *cuArray;
		cudaGraphicsSubResourceGetMappedArray(&cuArray, g_texture_cube->cudaResourceOutput, face, 0);
		ProcessCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_cube input) failed for eye [" + eyeName + "], face [" + std::to_string(face) + "]: ");

		// Copy pixels from cuArray (original texture) to allocated cudaLinearMemory
		cudaMemcpy2DFromArray(
			g_texture_cube->cudaLinearMemoryInput[face],	//dst location
			g_texture_cube->pitch,							//dst pitch
			cuArray,										//src array
			0, 0,											//w and h offset
			g_texture_cube->size * 4, g_texture_cube->size, //extent
			cudaMemcpyDeviceToDevice						//kind
		);
		ProcessCudaError("cudaMemcpy2DFromArray (cuda_texture_cube input) failed for eye [" + eyeName + "], face [" + std::to_string(face) + "]: ");
	}
}


// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

// Set up D3D context, and reset texture pointer (from Unity) to new texture data.
// Need to map / unmap cudaResource to allow access
static void DoRendering(std::string eyeName)
{
	struct TextureCube * g_texture_cube = GetTextureCubeForEye(eyeName);

	if (s_DeviceType == kUnityGfxRendererD3D11)
	{
		ID3D11DeviceContext* ctx = NULL;
		g_D3D11Device->GetImmediateContext(&ctx);

		if (g_texture_cube->cudaResourceOutput) {

			cudaStream_t    stream = 0;
			const int nbResources = 1;
			cudaGraphicsResource *ppResources[nbResources] =
			{
				g_texture_cube->cudaResourceOutput
			};

			cudaGraphicsMapResources(nbResources, ppResources, stream);
			getLastCudaError("cudaGraphicsMapResources(3) failed");

			// Run kernels which will populate the contents of those textures
			RunTextureFillingKernels(eyeName);

			// Unmap the resources
			cudaGraphicsUnmapResources(nbResources, ppResources, stream);
			getLastCudaError("cudaGraphicsUnmapResources(3) failed");
		}
		ctx->Release();
	}
}

// --------------------------------------------------------------------------
// Helpers (that can't go in Utils)
// --------------------------------------------------------------------------

// Returns pointer to right texture cube, depending on eye
struct TextureCube * GetTextureCubeForEye(std::string eyeName) {
	if (eyeName == kLeftEyeName)
	{
		return &g_texture_cube_left;
	}
	else if (eyeName == kRightEyeName)
	{
		return &g_texture_cube_right;
	}
}