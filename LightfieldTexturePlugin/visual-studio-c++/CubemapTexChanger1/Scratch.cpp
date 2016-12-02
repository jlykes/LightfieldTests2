//// -------------------------------------------------------------------
//// UNUSED
//
//// Previous update texture render code (in "DoRendering")
//// update native texture from code
////if (g_TexturePointer)
////{
////	ID3D11Texture2D* d3dtex = (ID3D11Texture2D*)g_TexturePointer;
////	D3D11_TEXTURE2D_DESC desc;
////	d3dtex->GetDesc(&desc);
//
////	//cuda_test();
//
//
////	unsigned char* data = new unsigned char[desc.Width*desc.Height * 4];
////	//FillTextureFromCode1(desc.Width, desc.Height, desc.Width * 4, data);
////	//ctx->UpdateSubresource(d3dtex, 0, NULL, data, desc.Width * 4, 0);
////	delete[] data;
////}
//
//// Load image to use as texture, and apply gradual brighteness increase / decrease
//// depending on how much time has passed
//static void FillTextureFromCode(int width, int height, int stride, unsigned char* dst)
//{
//	//if based texture not loaded
//	//__load it
//	//__convert it to unsigned char* ___baseData
//
//	//set ___brightnessMultiplier (function of time)
//
//	//__fillDst based on multiplying brightnessMultiplier * baseData value
//
//}
//
//
//static void FillTextureFromCode1(int width, int height, int stride, unsigned char* dst)
//{
//	const float t = g_Time * 4.0f;
//
//	//for (int y = 0; y < height; ++y)
//	//{
//	//	unsigned char* ptr = dst;
//	//	for (int x = 0; x < width; ++x)
//	//	{
//	//		DecideColor1(ptr);
//
//	//		// To next pixel (our pixels are 4 bpp)
//	//		ptr += 4;
//	//	}
//
//	//	// To next image row
//	//	dst += stride;
//	//}
//}
//
//// Decide what color all pixels should be depending on how much time has passed
//static void DecideColor1(unsigned char* ptr)
//{
//	int colorIndex = calcColorIndex();
//
//	ptr[0] = 0; //R
//	ptr[1] = 0; //G
//	ptr[2] = 0; //B
//	ptr[3] = 255; //A
//
//	switch (colorIndex) {
//	case 0:
//		ptr[0] = 255;
//		break;
//	case 1:
//		ptr[1] = 255;
//		break;
//	case 2:
//		ptr[2] = 255;
//	}
//}
//
//// Calculate which color index (0 = red, 1 = blue, 2 = green) to use depending on 
//// how much time has passed
//static int calcColorIndex()
//{
//	int timeInt = int(g_Time);
//	int secondInterval = 2;
//
//	return (roundDown(timeInt, secondInterval) / secondInterval) % 3;
//}
//
//
//// Round number down to given multiple
//static int roundDown(int number, int multiple)
//{
//	if (multiple == 0)
//		return number;
//
//	int remainder = number % multiple;
//	if (remainder == 0)
//		return number;
//
//	return number - remainder;
//}
