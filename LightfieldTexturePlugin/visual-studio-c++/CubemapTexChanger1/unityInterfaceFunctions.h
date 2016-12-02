/* ==========================================================================
   unityInterfaceFunctions.h
   ==========================================================================

   MAIN file - holds definition/implementation for all functions that link
   directly to Unity

*/

#pragma once

#include "Unity/IUnityInterface.h"
#include "Unity/IUnityGraphics.h"
#include "unityInterfaceFunctions.h"
#include "utils.h"
#include <string>

// Which platform we are on?
#if _MSC_VER
#define UNITY_WIN 1
#elif defined(__APPLE__)
#if defined(__arm__)
#define UNITY_IPHONE 1
#else
#define UNITY_OSX 1
#endif
#elif defined(UNITY_METRO) || defined(UNITY_ANDROID) || defined(UNITY_LINUX)
// these are defined externally
#else
#error "Unknown platform!"
#endif

// Which graphics device APIs we possibly support?
#if UNITY_METRO
#define SUPPORT_D3D11 1
#if WINDOWS_UWP
#define SUPPORT_D3D12 1
#endif
#elif UNITY_WIN
#define SUPPORT_D3D9 1
#define SUPPORT_D3D11 1 // comment this out if you don't have D3D11 header/library files
#ifdef _MSC_VER
#if _MSC_VER >= 1900
#define SUPPORT_D3D12 1
#endif
#endif
#define SUPPORT_OPENGL_LEGACY 1
#define SUPPORT_OPENGL_UNIFIED 1
#define SUPPORT_OPENGL_CORE 1
#elif UNITY_IPHONE || UNITY_ANDROID
#define SUPPORT_OPENGL_UNIFIED 1
#define SUPPORT_OPENGL_ES 1
#elif UNITY_OSX || UNITY_LINUX
#define SUPPORT_OPENGL_LEGACY 1
#define SUPPORT_OPENGL_UNIFIED 1
#define SUPPORT_OPENGL_CORE 1
#endif

// --------------------------------------------------------------------------
// Types
// --------------------------------------------------------------------------


// Data structure for cube texture shared between DX10 and CUDA
struct TextureCube
{
	ID3D11Texture2D         *pTexture;
	ID3D11ShaderResourceView *pSRView;
	cudaGraphicsResource    *cudaResource;
	void                    *cudaLinearMemory;
	size_t                  pitch;
	int                     size;
};

// --------------------------------------------------------------------------
// Global variables
// --------------------------------------------------------------------------

// UnitySetInterfaces
static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;

// Direct3D 11 setup/teardown code
static ID3D11Device* g_D3D11Device = NULL;

// Set time / streaming assets path
static float g_Time;
static std::string s_UnityStreamingAssetsPath;

// Textures
struct TextureCube g_texture_cube_left, g_texture_cube_right;
//struct TextureCube g_texture_cube;


// --------------------------------------------------------------------------
// Function Prototypes
// --------------------------------------------------------------------------

// Unity interface functions
static void DoEventGraphicsDeviceD3D11(UnityGfxDeviceEventType eventType);
static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);
static void UNITY_INTERFACE_API OnRenderEvent(int eventID);

// Texture / render handling
static void SetTextureFromUnityImplementation(void* texturePtr, std::string eyeName);
static void DoRendering(std::string eyeName);
void RunTextureFillingKernels(std::string eyeName);

// CUDA function calls
extern "C"
{
	void CudaWrapperHelloWorld();
	void CudaWrapperTextureCubeStrobelight(void *surface, int width, int height, size_t pitch, int face, float t);
}