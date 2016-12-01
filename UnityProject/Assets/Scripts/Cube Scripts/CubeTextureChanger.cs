using UnityEngine;
using System.Collections;

public class CubeTextureChanger : MonoBehaviour {

    public Texture preloadedTexture;
    bool textureFinishedLoading;
    bool usingPreloadedTexture;
    WWW w;
    Texture loadedTexture;
    Renderer rend;

    // Use this for initialization
    void Start () {
        usingPreloadedTexture = false;
        textureFinishedLoading = false;
        rend = GetComponent<Renderer>();
        string texturePath = "file:///" + Application.dataPath + "/Resources/cube_texture_1.png";
        w = new WWW(texturePath);
    }
	
	// Update is called once per frame
	void Update () {
        if (w.isDone && !textureFinishedLoading)
        {
            //Debug.Log("Done");
            loadedTexture = w.texture;
            textureFinishedLoading = true;
        }


        //if (Input.GetKeyDown(KeyCode.L))
        //{
        //    if (!usingPreloadedTexture)
        //    {
        //        rend.material.SetTexture("_MainTex", loadedTexture);
        //        usingPreloadedTexture = true;
        //    }
        //    else
        //    {
        //        rend.material.SetTexture("_MainTex", preloadedTexture);
        //        usingPreloadedTexture = false;
        //    }
        //}
	
	}
}
