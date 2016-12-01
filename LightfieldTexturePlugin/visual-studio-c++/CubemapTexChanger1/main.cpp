// Example low level rendering Unity plugin
#include "main.h"
#include "Unity/IUnityGraphics.h"

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
// Helper utilities


// Prints a string
//static void DebugLog(const char* str)
//{
//#if UNITY_WIN
//	OutputDebugStringA (str);
//#else
//	printf("%s", str);
//#endif
//}


//Setup for making Debug.Log in Unity callable from here
typedef void(__stdcall * DebugCallback) (const char *);
DebugCallback gDebugCallback;

extern "C" void __declspec(dllexport) RegisterDebugCallback(DebugCallback callback)
{
	gDebugCallback = callback;
}

void DebugInUnity(std::string message)
{
	gDebugCallback(message.c_str()); 
}

// COM-like Release macro
#ifndef SAFE_RELEASE
#define SAFE_RELEASE(a) if (a) { a->Release(); a = NULL; }
#endif



// --------------------------------------------------------------------------
// SetTimeFromUnity, an example function we export which is called by one of the scripts.

static float g_Time;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTimeFromUnity(float t) { g_Time = t; }



// --------------------------------------------------------------------------
// SetTextureFromUnity, an example function we export which is called by one of the scripts.

static void* g_TexturePointer = NULL;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* texturePtr)
{
	// A script calls this at initialization time; just remember the texture pointer here.
	// Will update texture pixels each frame from the plugin rendering event (texture update
	// needs to happen on the rendering thread).
	g_TexturePointer = texturePtr;
}


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


// -------------------------------------------------------------------
//  For filling textures


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


// Set up D3D context, and reset texture pointer (from Unity) to new texture data
static void DoRendering()
{
#if SUPPORT_D3D11
	// D3D11 case
	if (s_DeviceType == kUnityGfxRendererD3D11)
	{
		ID3D11DeviceContext* ctx = NULL;
		g_D3D11Device->GetImmediateContext(&ctx);

		// update native texture from code
		if (g_TexturePointer)
		{
			ID3D11Texture2D* d3dtex = (ID3D11Texture2D*)g_TexturePointer;
			D3D11_TEXTURE2D_DESC desc;
			d3dtex->GetDesc(&desc);

			cuda_test();


			unsigned char* data = new unsigned char[desc.Width*desc.Height * 4];
			//FillTextureFromCode1(desc.Width, desc.Height, desc.Width * 4, data);
			//ctx->UpdateSubresource(d3dtex, 0, NULL, data, desc.Width * 4, 0);
			delete[] data;
		}

		ctx->Release();
	}
#endif
}




// -------------------------------------------------------------------
// Helpers for trivial color change 1


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
