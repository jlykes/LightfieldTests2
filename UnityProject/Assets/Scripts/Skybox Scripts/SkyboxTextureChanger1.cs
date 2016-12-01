using UnityEngine;
using System.Collections;
using UnityEngine.UI;

public class SkyboxTextureChanger1 : MonoBehaviour {

    public Texture texture1;
    public Texture texture2;
    WWW w;
    Cubemap loadedTextureAsCubemap;
    Texture2D loadedTexture;
    public string loadedTextureName;
    public string eyeName;
    string textureName;
    bool usingPreloadedTexture;
    bool textureFinishedLoading;
    Renderer rend;


	// Use this for initialization
	void Start () {
        textureName = "tex1";
        rend = GetComponent<Renderer>();
        usingPreloadedTexture = false;
        textureFinishedLoading = false;
        string texturePath = "file:///" + Application.dataPath + "/Resources/" + eyeName + "-" + loadedTextureName + ".png";
        w = new WWW(texturePath);
    }
	
	// Update is called once per frame
	void Update () {
        if (w.isDone && !textureFinishedLoading)
        {
            Debug.Log("Texture finished loading.");
            mapTextureToCubemap();
            textureFinishedLoading = true;
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log(Application.dataPath);
            if (textureName == "tex1")
            {
                rend.material.SetTexture("_Tex", texture2);
                textureName = "tex2";
            } else
            {
                rend.material.SetTexture("_Tex", texture1);
                textureName = "tex1";
            }
        }

        if (Input.GetKeyDown(KeyCode.L))
        {
            if (!textureFinishedLoading) Debug.Log("Texture not finished loading");
            if (!usingPreloadedTexture && textureFinishedLoading) 
            {
                rend.material.SetTexture("_Tex", loadedTextureAsCubemap);
                usingPreloadedTexture = true;
            }
            else
            {
                rend.material.SetTexture("_Tex", texture1);
                textureName = "tex1";
                usingPreloadedTexture = false;
            }
        }


	}


    // Take loaded texture, and map to cube map
    void mapTextureToCubemap()
    {
        loadedTexture = w.texture;

        TextureScaler.scale(loadedTexture, 1024 * 6, 1024);
        loadedTexture = FlipTextureY(loadedTexture);

        int faceWidth = loadedTexture.width / 6;
        int faceHeight = loadedTexture.height;

        loadedTextureAsCubemap = new Cubemap(faceWidth, TextureFormat.RGB24, false);

        string[] cubemapFaceTypeNames = System.Enum.GetNames(typeof(CubemapFace));

        for (int i = 0; i < cubemapFaceTypeNames.Length - 1; i++)
        {
            Color[] cubeMapFace = loadedTexture.GetPixels(i * faceWidth, 0, faceWidth, faceHeight);
            loadedTextureAsCubemap.SetPixels(cubeMapFace, (CubemapFace)i);
            loadedTextureAsCubemap.Apply();
        }

        loadedTextureAsCubemap.SmoothEdges(100);
        loadedTextureAsCubemap.Apply();
    }



    Texture2D FlipTextureY(Texture2D original)
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
}
