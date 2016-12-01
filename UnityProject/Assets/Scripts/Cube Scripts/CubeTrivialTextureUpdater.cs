using UnityEngine;
using System.Collections;
using System;
using System.Runtime.InteropServices;

public class CubeTrivialTextureUpdater : MonoBehaviour {

    [DllImport("TrivialTexChanger")]
    private static extern void SetTimeFromUnity(float t);

    [DllImport("TrivialTexChanger")]
    private static extern void SetTextureFromUnity(System.IntPtr texture);

    [DllImport("TrivialTexChanger")]
    private static extern IntPtr GetRenderEventFunc();


    IEnumerator Start()
    {
        CreateTextureAndPassToPlugin();
        yield return StartCoroutine("CallPluginAtEndOfFrames");
    }


    private void CreateTextureAndPassToPlugin()
    {
        // Create a texture
        Texture2D tex = new Texture2D(256, 256, TextureFormat.ARGB32, false);
        // Set point filtering just so we can see the pixels clearly
        tex.filterMode = FilterMode.Point;
        // Call Apply() so it's actually uploaded to the GPU
        tex.Apply();

        // Set texture onto our matrial
        GetComponent<Renderer>().material.mainTexture = tex;

        // Pass texture pointer to the plugin
        SetTextureFromUnity(tex.GetNativeTexturePtr());
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
        }
    }

	
	// Update is called once per frame
	void Update () {
	
	}
}
