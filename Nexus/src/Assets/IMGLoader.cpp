#define STB_IMAGE_IMPLEMENTATION
#include "IMGLoader.h"
#include "stb_image.h"

IMGLoader::IMGLoader()
{
}

IMGLoader::~IMGLoader()
{
}

Texture IMGLoader::LoadIMG(const std::string& filepath)
{
	int width, height, channels;

	void* pixels;
	bool HDR = false;
	if (stbi_is_hdr(filepath.c_str()))
	{
		HDR = true;
		pixels = stbi_loadf(filepath.c_str(), &width, &height, &channels, 4);
	}
	else
		pixels = stbi_load(filepath.c_str(), &width, &height, &channels, 4);

	if (pixels == nullptr)
		std::cout << "IMGLoader: Failed to load texture " << filepath << std::endl;

	return Texture(width, height, channels, HDR, pixels);
}

Texture IMGLoader::LoadIMG(const aiTexture* texture)
{
	int width, height, channels;
	unsigned char* pixels = stbi_load_from_memory((const stbi_uc*)texture->pcData, texture->mWidth, &width, &height, &channels, 4);

	if (pixels == nullptr)
		std::cout << "IMGLoader: Failed to load an embedded texture" << std::endl;

	return Texture(width, height, channels, false, pixels);
}


