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


// -------------------------------------------------------------------------------------------------------
// =======================================================================================================
// TEXTURE / RENDER HANDLING
// =======================================================================================================
// -------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Texture setup / filling
// --------------------------------------------------------------------------

// Implementation of texture setting function that Unity calls
static void SetTextureFromUnityImplementation(void* texturePtr, std::string eyeName) {

	struct TextureCube * g_texture_cube = GetTextureCubeForEye(eyeName);

	// Initialize g_texture_cube to by Unity texture; set other parameters
	g_texture_cube->pTexture = (ID3D11Texture2D*)texturePtr;
	D3D11_TEXTURE2D_DESC desc;
	g_texture_cube->pTexture->GetDesc(&desc);
	g_texture_cube->size = desc.Width;

	// Puts pTexture into cudaResource
	cudaGraphicsD3D11RegisterResource(&g_texture_cube->cudaResource, g_texture_cube->pTexture, cudaGraphicsRegisterFlagsNone);
	ProcessCudaError("cudaGraphicsD3D11RegisterResource (g_texture_cube) failed: ");

	// Create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
	cudaMallocPitch(&g_texture_cube->cudaLinearMemory, &g_texture_cube->pitch, g_texture_cube->size * 4, g_texture_cube->size);
	ProcessCudaError("cudaMallocPitch (g_texture_cube) failed: ");
	cudaMemset(g_texture_cube->cudaLinearMemory, 128, g_texture_cube->pitch * g_texture_cube->size);
	ProcessCudaError("cudaMemset (g_texture_cube) failed: ");
}


//Run kernels in CUDA file
void RunTextureFillingKernels(std::string eyeName)
{
	struct TextureCube * g_texture_cube = GetTextureCubeForEye(eyeName);

	static float t = 0.0f;
	for (int face = 0; face < 6; ++face)
	{
		cudaArray *cuArray;
		cudaGraphicsSubResourceGetMappedArray(&cuArray, g_texture_cube->cudaResource, face, 0);
		ProcessCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_cube) failed: ");

		// Kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
		CudaWrapperTextureCubeStrobelight(g_texture_cube->cudaLinearMemory, g_texture_cube->size, g_texture_cube->size, g_texture_cube->pitch, face, t);
		ProcessCudaError("cuda_texture_cube failed: ");

		// Then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
		cudaMemcpy2DToArray(
			cuArray, // dst array
			0, 0,    // offset
			g_texture_cube->cudaLinearMemory, g_texture_cube->pitch, // src
			g_texture_cube->size * 4, g_texture_cube->size,            // extent
			cudaMemcpyDeviceToDevice); // kind
		ProcessCudaError("cudaMemcpy2DToArray failed: ");
	}
	t += 0.1f;
}


// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

// Set up D3D context, and reset texture pointer (from Unity) to new texture data
static void DoRendering(std::string eyeName)
{
	struct TextureCube * g_texture_cube = GetTextureCubeForEye(eyeName);

	if (s_DeviceType == kUnityGfxRendererD3D11)
	{
		ID3D11DeviceContext* ctx = NULL;
		g_D3D11Device->GetImmediateContext(&ctx);

		if (g_texture_cube->cudaResource) {

			cudaStream_t    stream = 0;
			const int nbResources = 1;
			cudaGraphicsResource *ppResources[nbResources] =
			{
				g_texture_cube->cudaResource
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