// Example low level rendering Unity plugin
#include "main.h"
#include "Unity/IUnityGraphics.h"
#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

// --------------------------------------------------------------------------
// Include headers for the graphics APIs we support


#if SUPPORT_D3D11
#	include <d3d11.h>
#	include "Unity/IUnityGraphicsD3D11.h"
#endif


// --------------------------------------------------------------------------
// Include stuff for CUDA

#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h



// --------------------------------------------------------------------------
// Texture struct

// Data structure for cube texture shared between DX10 and CUDA
struct
{
	ID3D11Texture2D         *pTexture;
	ID3D11ShaderResourceView *pSRView;
	cudaGraphicsResource    *cudaResource;
	void                    *cudaLinearMemory;
	size_t                  pitch;
	int                     size;
} g_texture_cube;


// --------------------------------------------------------------------------
// SetTimeFromUnity, an example function we export which is called by one of the scripts.

static float g_Time;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTimeFromUnity(float t) { g_Time = t; }


// --------------------------------------------------------------------------
// SetUnityStreamingAssetsPath, an example function we export which is called by one of the scripts.

static std::string s_UnityStreamingAssetsPath;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetUnityStreamingAssetsPath(const char* path)
{
	s_UnityStreamingAssetsPath = path;
}


// --------------------------------------------------------------------------
// UnitySetInterfaces

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);

static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;

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
// Graphics pointer
static void* g_TexturePointer = NULL;

// --------------------------------------------------------------------------
// GraphicsDeviceEvent

#if SUPPORT_D3D11
static void DoEventGraphicsDeviceD3D11(UnityGfxDeviceEventType eventType);
#endif

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
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
		g_TexturePointer = NULL;
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

#if SUPPORT_D3D11
	//This seems to be the only reason we care about all of the graphics device stuff - 
	//so we can tell whether running DirectX, etc. and use that to define what our code
	//should look like
	if (currentDeviceType == kUnityGfxRendererD3D11)
		DoEventGraphicsDeviceD3D11(eventType);
#endif

}


// --------------------------------------------------------------------------
// OnRenderEvent
// This will be called for GL.IssuePluginEvent script calls; eventID will
// be the integer passed to IssuePluginEvent. In this example, we just ignore
// that value.

static void DoRendering();

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
	// Unknown graphics device type? Do nothing.
	if (s_DeviceType == kUnityGfxRendererNull)
		return;

	DoRendering();
}

// --------------------------------------------------------------------------
// GetRenderEventFunc, an example function we export which is used to get a rendering event callback function.
extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
	return OnRenderEvent;
}


// -------------------------------------------------------------------
//  Direct3D 11 setup/teardown code


#if SUPPORT_D3D11

static ID3D11Device* g_D3D11Device = NULL;

static void DoEventGraphicsDeviceD3D11(UnityGfxDeviceEventType eventType)
{
	if (eventType == kUnityGfxDeviceEventInitialize)
	{
		IUnityGraphicsD3D11* d3d11 = s_UnityInterfaces->Get<IUnityGraphicsD3D11>();
		g_D3D11Device = d3d11->GetDevice();
	}
}

#endif // #if SUPPORT_D3D11


// --------------------------------------------------------------------------
// SetTextureFromUnity, an example function we export which is called by one of the scripts.


extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* texturePtr)
{
	// Initialize g_texture_cube to by Unity texture; set other parameters
	g_texture_cube.pTexture = (ID3D11Texture2D*)texturePtr;
	D3D11_TEXTURE2D_DESC desc;
	g_texture_cube.pTexture->GetDesc(&desc);
	g_texture_cube.size = desc.Width;

	// Puts pTexture into cudaResource
	cudaGraphicsD3D11RegisterResource(&g_texture_cube.cudaResource, g_texture_cube.pTexture, cudaGraphicsRegisterFlagsNone);
	ProcessCudaError("cudaGraphicsD3D11RegisterResource (g_texture_cube) failed: ");

	// Create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
	cudaMallocPitch(&g_texture_cube.cudaLinearMemory, &g_texture_cube.pitch, g_texture_cube.size * 4, g_texture_cube.size);
	ProcessCudaError("cudaMallocPitch (g_texture_cube) failed: ");
	cudaMemset(g_texture_cube.cudaLinearMemory, 128, g_texture_cube.pitch * g_texture_cube.size);
	ProcessCudaError("cudaMemset (g_texture_cube) failed: ");
}


// -------------------------------------------------------------------
//  For filling textures

//Run kernels in CUDA file
void RunKernels()
{
	static float t = 0.0f;
	for (int face = 0; face < 6; ++face)
	{
		cudaArray *cuArray;
		cudaGraphicsSubResourceGetMappedArray(&cuArray, g_texture_cube.cudaResource, face, 0);
		ProcessCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_cube) failed: ");

		// kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
		cuda_texture_cube(g_texture_cube.cudaLinearMemory, g_texture_cube.size, g_texture_cube.size, g_texture_cube.pitch, face, t);
		ProcessCudaError("cuda_texture_cube failed: ");

		//then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
		cudaMemcpy2DToArray(
			cuArray, // dst array
			0, 0,    // offset
			g_texture_cube.cudaLinearMemory, g_texture_cube.pitch, // src
			g_texture_cube.size * 4, g_texture_cube.size,            // extent
			cudaMemcpyDeviceToDevice); // kind
		ProcessCudaError("cudaMemcpy2DToArray failed: ");
	}
	t += 0.1f;
	DebugInUnity("Kernel ran again");
}


// Set up D3D context, and reset texture pointer (from Unity) to new texture data
static void DoRendering()
{
#if SUPPORT_D3D11
	// D3D11 case
	if (s_DeviceType == kUnityGfxRendererD3D11)
	{
		ID3D11DeviceContext* ctx = NULL;
		g_D3D11Device->GetImmediateContext(&ctx);

		if (g_texture_cube.cudaResource) {
			
			cudaStream_t    stream = 0;
			const int nbResources = 1;
			cudaGraphicsResource *ppResources[nbResources] =
			{
				g_texture_cube.cudaResource
			};

			cudaGraphicsMapResources(nbResources, ppResources, stream);
			getLastCudaError("cudaGraphicsMapResources(3) failed");

			//
			// run kernels which will populate the contents of those textures
			//
			RunKernels();

			//
			// unmap the resources
			//
			cudaGraphicsUnmapResources(nbResources, ppResources, stream);
			getLastCudaError("cudaGraphicsUnmapResources(3) failed");
		}

		ctx->Release();
	}
#endif
}




// -------------------------------------------------------------------
// UNUSED

// Previous update texture render code (in "DoRendering")
// update native texture from code
//if (g_TexturePointer)
//{
//	ID3D11Texture2D* d3dtex = (ID3D11Texture2D*)g_TexturePointer;
//	D3D11_TEXTURE2D_DESC desc;
//	d3dtex->GetDesc(&desc);

//	//cuda_test();


//	unsigned char* data = new unsigned char[desc.Width*desc.Height * 4];
//	//FillTextureFromCode1(desc.Width, desc.Height, desc.Width * 4, data);
//	//ctx->UpdateSubresource(d3dtex, 0, NULL, data, desc.Width * 4, 0);
//	delete[] data;
//}

// Load image to use as texture, and apply gradual brighteness increase / decrease
// depending on how much time has passed
static void FillTextureFromCode(int width, int height, int stride, unsigned char* dst)
{
	//if based texture not loaded
	//__load it
	//__convert it to unsigned char* ___baseData

	//set ___brightnessMultiplier (function of time)

	//__fillDst based on multiplying brightnessMultiplier * baseData value

}


static void FillTextureFromCode1(int width, int height, int stride, unsigned char* dst)
{
	const float t = g_Time * 4.0f;

	//for (int y = 0; y < height; ++y)
	//{
	//	unsigned char* ptr = dst;
	//	for (int x = 0; x < width; ++x)
	//	{
	//		DecideColor1(ptr);

	//		// To next pixel (our pixels are 4 bpp)
	//		ptr += 4;
	//	}

	//	// To next image row
	//	dst += stride;
	//}
}

// Decide what color all pixels should be depending on how much time has passed
static void DecideColor1(unsigned char* ptr)
{
	int colorIndex = calcColorIndex();

	ptr[0] = 0; //R
	ptr[1] = 0; //G
	ptr[2] = 0; //B
	ptr[3] = 255; //A

	switch (colorIndex) {
	case 0:
		ptr[0] = 255;
		break;
	case 1:
		ptr[1] = 255;
		break;
	case 2:
		ptr[2] = 255;
	}
}

// Calculate which color index (0 = red, 1 = blue, 2 = green) to use depending on 
// how much time has passed
static int calcColorIndex()
{
	int timeInt = int(g_Time);
	int secondInterval = 2;

	return (roundDown(timeInt, secondInterval) / secondInterval) % 3;
}


// Round number down to given multiple
static int roundDown(int number, int multiple)
{
	if (multiple == 0)
		return number;

	int remainder = number % multiple;
	if (remainder == 0)
		return number;

	return number - remainder;
}
