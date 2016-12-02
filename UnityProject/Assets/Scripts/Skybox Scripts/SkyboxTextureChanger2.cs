using UnityEngine;
using System.Collections;
using System;
using System.Runtime.InteropServices;

public class SkyboxTextureChanger2 : MonoBehaviour {

    // ----------------------------------------------------------------------------
    // Public Global Parameters
    // ----------------------------------------------------------------------------

    public string loadedTextureName;
    public string eyeName;


    // ----------------------------------------------------------------------------
    // Private Global Variables
    // ----------------------------------------------------------------------------

    WWW w;
    Renderer rend;
    Texture2D textureAs2DImg;
    Cubemap textureAsCubemap;
    int pluginEventNumber;


    // ----------------------------------------------------------------------------
    // Constants
    // ----------------------------------------------------------------------------

    int kTextureSegmentWidth = 1024;
    int kTextureSegmentHeight = 1024;
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
    IEnumerator loadTextureFile()
    {
        rend = GetComponent<Renderer>();
        string texturePath = "file:///" + Application.dataPath + "/Resources/" + eyeName + "-" + loadedTextureName + ".png";

        w = new WWW(texturePath);
        yield return w;

        getLoadedTextureAs2D();
        convertToCubemapAndSetInPlugin();
    }


    private void getLoadedTextureAs2D()
    {
        // Get and scale and flip texture from loader WWW
        textureAs2DImg = w.texture;

        TextureScaler.scale(textureAs2DImg, 
                            kTextureSegmentWidth * kNumCubemapSides, 
                            kTextureSegmentHeight);
    
        textureAs2DImg = FlipTextureY(textureAs2DImg);

        // Set point filtering just so we can see the pixels clearly
        textureAs2DImg.filterMode = FilterMode.Point;

        // Call Apply() so it's actually uploaded to the GPU
        textureAs2DImg.Apply();
    }

    // Creates cubemap holder, puts loaded texture in, and sets pointer in plugin
    private void convertToCubemapAndSetInPlugin()
    {
        textureAsCubemap = new Cubemap(kTextureSegmentWidth, TextureFormat.RGB24, false);

        string[] cubemapFaceTypeNames = System.Enum.GetNames(typeof(CubemapFace));

        for (int i = 0; i < cubemapFaceTypeNames.Length - 1; i++)
        {
            Color[] cubeMapFace = textureAs2DImg.GetPixels(i * kTextureSegmentWidth, 0, kTextureSegmentWidth, kTextureSegmentHeight);
            textureAsCubemap.SetPixels(cubeMapFace, (CubemapFace)i);
            textureAsCubemap.Apply();
        }

        textureAsCubemap.SmoothEdges(100);
        textureAsCubemap.Apply();
        rend.material.SetTexture("_Tex", textureAsCubemap);

        SetTextureFromUnity(textureAsCubemap.GetNativeTexturePtr(), eyeName);
    }

    // Flip texture (to fix issue with mysteriously importing with wrong orientation)
    private Texture2D FlipTextureY(Texture2D original)
    {
        Texture2D flipped = new Texture2D(original.width, original.height);

        int xN = original.width;
        int yN = original.height;


        for (int i = 0; i < xN; i++)
        {
            for (int j = 0; j < yN; j++)
            {
                flipped.SetPixel(i, yN - j - 1, original.GetPixel(i, j));
            }
        }
        flipped.Apply();

        return flipped;
    }


    // ----------------------------------------------------------------------------
    // Main Loop
    // ----------------------------------------------------------------------------

    IEnumerator Start () {
        //Make it so that plugin can use Debug.Log
        RegisterDebugCallback(new DebugCallback(DebugMethod));
        //LogConsoleOutputToConsole();
        LogConsoleOutputToFile();

        //Set event nuber that plugin will use to determine eye
        pluginEventNumber = (eyeName == "L") ? 1 : 2;

        //Load texture file, and then start call to plugin at end of each frame
        StartCoroutine("loadTextureFile");
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
