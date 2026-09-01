#define STB_IMAGE_IMPLEMENTATION
#include "IMGLoader.h"
#include "stb_image.h"

namespace Nexus {

	IMGLoader::IMGLoader()
	{
	}

	IMGLoader::~IMGLoader()
	{
	}

	std::optional<Texture> IMGLoader::LoadIMG(const std::string& filepath, Texture::Type type)
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

		// Returning null rather than an empty Texture: stb writes width/height only on success, so
		// a Texture built from a failed load carries uninitialised dimensions that later reach
		// cudaMallocArray. One representation of failure, checked at the call site.
		if (pixels == nullptr)
		{
			std::cout << "IMGLoader: Failed to load texture " << filepath << std::endl;
			return std::nullopt;
		}

		return Texture(width, height, channels, HDR, type, StbImageData(pixels));
	}

	std::optional<Texture> IMGLoader::LoadIMG(const aiTexture* texture, Texture::Type type)
	{
		// mHeight != 0 means the embedded texture is raw ARGB rather than an encoded file, which
		// stbi_load_from_memory cannot read.
		if (texture->mHeight != 0)
		{
			std::cout << "IMGLoader: Unsupported uncompressed embedded texture" << std::endl;
			return std::nullopt;
		}

		int width, height, channels;
		unsigned char* pixels = stbi_load_from_memory((const stbi_uc*)texture->pcData, texture->mWidth, &width, &height, &channels, 4);

		if (pixels == nullptr)
		{
			std::cout << "IMGLoader: Failed to load an embedded texture" << std::endl;
			return std::nullopt;
		}

		return Texture(width, height, channels, false, type, StbImageData(pixels));
	}

}
