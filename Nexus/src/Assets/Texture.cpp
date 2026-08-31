#include "Texture.h"
#include <cstdint>
#include <utility>
#include "stb_image.h"


namespace Nexus {

	void StbImageDeleter::operator()(void* pixels) const
	{
		stbi_image_free(pixels);
	}

	Texture::Texture(uint32_t w, uint32_t h, uint32_t c, bool isHDR, StbImageData d)
		: width(w), height(h), channels(c), HDR(isHDR), pixels(std::move(d))
	{
	}

	void Texture::UploadToDevice()
	{
		deviceTexture = CudaTexture(pixels.get(), width, height, HDR, sRGB);
	}

}
