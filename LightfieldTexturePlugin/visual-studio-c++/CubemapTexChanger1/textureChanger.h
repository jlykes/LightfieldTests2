/* ==========================================================================
   textureChanger.h
   ==========================================================================

   Implementation of functions to do with rendering / texturing (may be set
   up with wrappers in the unityInterfaceFunctions file so that Unity can
   call)

*/

#pragma once

// --------------------------------------------------------------------------
// General include headers
// --------------------------------------------------------------------------

#include "Unity/IUnityGraphics.h"
#include "unityInterfaceFunctions.h"
#include "utils.h"
#include "unityPluginSetup.h"

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


