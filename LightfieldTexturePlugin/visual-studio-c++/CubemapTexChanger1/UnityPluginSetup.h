/* ==========================================================================
   UnityPluginSetup.h
   ==========================================================================

   Implementation of some basic Unity Plugin setup functions (e.g. set time)

*/

#pragma once

#include "utils.h"

#if SUPPORT_D3D11
#	include <d3d11.h>
#	include "Unity/IUnityGraphicsD3D11.h"
#endif

#include "Unity/IUnityGraphics.h"
#include <string>



// Plugin setup shit









// --------------------------------------------------------------------------
// SetTimeFromUnity, an example function we export which is called by one of 
// the scripts.
// --------------------------------------------------------------------------

//extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTimeFromUnity(float t);



// --------------------------------------------------------------------------
// SetUnityStreamingAssetsPath, an example function we export which is called 
// by one of the scripts.
// --------------------------------------------------------------------------


//extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetUnityStreamingAssetsPath(const char* path);


// --------------------------------------------------------------------------
// UnitySetInterfaces
// --------------------------------------------------------------------------













// --------------------------------------------------------------------------
// GraphicsDeviceEvent
// --------------------------------------------------------------------------










// --------------------------------------------------------------------------
//  Direct3D 11 setup/teardown code
// --------------------------------------------------------------------------



