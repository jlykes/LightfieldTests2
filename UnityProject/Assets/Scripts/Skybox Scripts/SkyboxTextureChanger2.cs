using UnityEngine;
using System.Collections;
using System;
using System.Runtime.InteropServices;

public class SkyboxTextureChanger2 : MonoBehaviour {

    // ----------------------------------------------------------------------------
    // Public Global Parameters

    public string loadedTextureName;
    public string eyeName;


    // ----------------------------------------------------------------------------
    // Private Global Variables

    WWW w;
    Renderer rend;
    bool textureFinishedLoading;
    Texture2D textureAs2DImg;
    Cubemap textureAsCubemap;


    // ----------------------------------------------------------------------------
    // Constants

    int kTextureSegmentWidth = 1024;
    int kTextureSegmentHeight = 1024;
    int kNumCubemapSides = 6;

    // ----------------------------------------------------------------------------
    // Plugin Function Prototypes

    private delegate void DebugCallback(string message);

    [DllImport("CubemapTexChanger1")]
    private static extern void SetTimeFromUnity(float t);

    [DllImport("CubemapTexChanger1")]
    private static extern void SetTextureFromUnity(System.IntPtr texture);

    [DllImport("CubemapTexChanger1")]
    private static extern IntPtr GetRenderEventFunc();


    [DllImport("CubemapTexChanger1")]
    private static extern void RegisterDebugCallback(DebugCallback callback);

    // ----------------------------------------------------------------------------
    // Debug

    private static void DebugMethod(string message)
    {
        Debug.Log("CubemapTexChanger1: " + message);
    }


    // ----------------------------------------------------------------------------
    // Texture Manipulation

    // Load texture from file for first time
    IEnumerator loadTextureFile()
    {
        rend = GetComponent<Renderer>();
        string texturePath = "file:///" + Application.dataPath + "/Resources/" + eyeName + "-" + loadedTextureName + ".png";
        w = new WWW(texturePath);
        yield return w;
        Debug.Log("Called 'loadTextureFile'.");
    }


    private void GetLoadedTextureAndPassToPlugin()
    {
        textureAs2DImg = w.texture;

        TextureScaler.scale(textureAs2DImg, 
                            kTextureSegmentWidth * kNumCubemapSides, 
                            kTextureSegmentHeight);
    
        textureAs2DImg = FlipTextureY(textureAs2DImg);

        // Set point filtering just so we can see the pixels clearly
        textureAs2DImg.filterMode = FilterMode.Point;

        // Call Apply() so it's actually uploaded to the GPU
        textureAs2DImg.Apply();

        // Pass texture pointer to the plugin
        SetTextureFromUnity(textureAs2DImg.GetNativeTexturePtr());
        Debug.Log("Called 'GetLoadedTextureAndPassToPlugin'.");
    }


    // Take loaded texture, and map to cube map
    private void setPluginGenerated2DTextureAsCubemap()
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
        Debug.Log("Called 'setPluginGenerated2DTextureAsCubemap'.");
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
        Debug.Log("Called 'FlipTextureY'.");

        return flipped;
    }



    // ----------------------------------------------------------------------------
    // Main Loop

    IEnumerator Start () {
        RegisterDebugCallback(new DebugCallback(DebugMethod));

        StartCoroutine("loadTextureFile");
        Debug.Log("TextureFileLoaded");
        GetLoadedTextureAndPassToPlugin();
        setPluginGenerated2DTextureAsCubemap(); //May not need to always call this


        Debug.Log("Called 'Start'.");
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

            // Issue a plugin event with arbitrary integer identifier.
            // The plugin can distinguish between different
            // things it needs to do based on this ID.
            // For our simple plugin, it does not matter which ID we pass here.

            GL.IssuePluginEvent(GetRenderEventFunc(), 1);
            //setPluginGenerated2DTextureAsCubemap();
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}
