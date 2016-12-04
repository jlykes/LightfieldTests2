//This version imports the texture using the plugin, OUTSIDE OF UNITY, in .dds format

using UnityEngine;
using System.Collections;
using System;
using System.Runtime.InteropServices;

public class SkyboxTextureChanger3 : MonoBehaviour
{

    // ----------------------------------------------------------------------------
    // Public Global Parameters
    // ----------------------------------------------------------------------------

    public string eyeName;
    public Texture2D dummyTexAs2DImg;
    public float brightnessIntervalPeriod;

    // ----------------------------------------------------------------------------
    // Private Global Variables
    // ----------------------------------------------------------------------------

    Renderer rend;
    Cubemap textureAsCubemap;
    int pluginEventNumber;


    // ----------------------------------------------------------------------------
    // Constants
    // ----------------------------------------------------------------------------

    int kTextureSegmentWidth = 1024; //Need to update to be same as texture imported by plugin
    int kTextureSegmentHeight = 1024; //Need to update to be same as texture imported by plugin
    int kNumCubemapSides = 6;

    // ----------------------------------------------------------------------------
    // Plugin Function Prototypes
    // ----------------------------------------------------------------------------

    private delegate void DebugCallback(string message);

    [DllImport("CubemapTexChanger1")]
    private static extern void SetTimeFromUnity(float t);

    [DllImport("CubemapTexChanger1")]
    private static extern void SetTextureFromUnity(System.IntPtr texture, string eyeName);

    [DllImport("CubemapTexChanger1")]
    private static extern void SetBrightnessIntervalPeriodFromUnity(float brightnessIntervalPeriod);

    [DllImport("CubemapTexChanger1")]
    private static extern IntPtr GetRenderEventFunc();

    [DllImport("CubemapTexChanger1")]
    private static extern void RegisterDebugCallback(DebugCallback callback);

    [DllImport("CubemapTexChanger1")]
    private static extern void LogConsoleOutputToConsole();

    [DllImport("CubemapTexChanger1")]
    private static extern void LogConsoleOutputToFile();

    // ----------------------------------------------------------------------------
    // Debug
    // ----------------------------------------------------------------------------

    // Method that plugin can use to print to Unity console via Debug.Log
    private static void DebugMethod(string message)
    {
        Debug.Log("CubemapTexChanger1: " + message);
    }


    // ----------------------------------------------------------------------------
    // Texture Manipulation
    // ----------------------------------------------------------------------------

    // Load texture from file for first time
    void setTextureInPlugin()
    {
        rend = GetComponent<Renderer>();

        dummyTexAs2DImg.Resize(kTextureSegmentWidth * kNumCubemapSides,
                            kTextureSegmentHeight, TextureFormat.RGB24, false);

        convertToCubemapAndSetInPlugin();
    }

    // Creates cubemap holder, puts loaded texture in, and sets pointer in plugin
    private void convertToCubemapAndSetInPlugin()
    {
        textureAsCubemap = new Cubemap(kTextureSegmentWidth, TextureFormat.RGB24, false);

        string[] cubemapFaceTypeNames = System.Enum.GetNames(typeof(CubemapFace));

        for (int i = 0; i < cubemapFaceTypeNames.Length - 1; i++)
        {
            Color[] cubeMapFace = dummyTexAs2DImg.GetPixels(i * kTextureSegmentWidth, 0, kTextureSegmentWidth, kTextureSegmentHeight);
            textureAsCubemap.SetPixels(cubeMapFace, (CubemapFace)i);
            textureAsCubemap.Apply();
        }

        textureAsCubemap.SmoothEdges(100);
        textureAsCubemap.Apply();
        rend.material.SetTexture("_Tex", textureAsCubemap);

        SetTextureFromUnity(textureAsCubemap.GetNativeTexturePtr(), eyeName);
    }

    // ----------------------------------------------------------------------------
    // Main Loop
    // ----------------------------------------------------------------------------

    IEnumerator Start()
    {
        SetBrightnessIntervalPeriodFromUnity(brightnessIntervalPeriod);
        
        //Make it so that plugin can use Debug.Log
        RegisterDebugCallback(new DebugCallback(DebugMethod));
        //LogConsoleOutputToConsole();
        //LogConsoleOutputToFile();

        //Set event nuber that plugin will use to determine eye
        pluginEventNumber = (eyeName == "L") ? 1 : 2;

        //Load texture file, and then start call to plugin at end of each frame
        setTextureInPlugin();
        yield return StartCoroutine("CallPluginAtEndOfFrames");
    }

    private IEnumerator CallPluginAtEndOfFrames()
    {
        while (true)
        {
            // Wait until all frame rendering is done
            yield return new WaitForEndOfFrame();

            // Set time for the plugin
            SetTimeFromUnity(Time.timeSinceLevelLoad);

            // Issue a plugin event; ID tells which eye to use
            GL.IssuePluginEvent(GetRenderEventFunc(), pluginEventNumber);
        }
    }

}
