#include "Texture.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <cstdint>
#include "Utils/Utils.h"


namespace Nexus {

	Texture::Texture(uint32_t w, uint32_t h, uint32_t c, bool isHDR, void* d) : width(w), height(h), channels(c), HDR(isHDR), pixels(d)
	{
	}

	Texture::Texture(const Texture& other)
	{
		CopyFrom(other);
	}

	Texture::Texture(Texture&& other)
	{
		width = other.width;
		height = other.height;
		channels = other.channels;
		sRGB = other.sRGB;
		HDR = other.HDR;
		type = other.type;
		pixels = other.pixels;
		other.pixels = nullptr;
	}

	Texture& Texture::operator=(const Texture& other)
	{
		free(pixels);
		CopyFrom(other);
		return *this;
	}

	Texture& Texture::operator=(Texture&& other)
	{
		free(pixels);
		width = other.width;
		height = other.height;
		channels = other.channels;
		sRGB = other.sRGB;
		HDR = other.HDR;
		type = other.type;
		pixels = other.pixels;
		other.pixels = nullptr;
		return *this;
	}

	Texture::~Texture()
	{
		free(pixels);
	}

	cudaTextureObject_t Texture::ToDevice(const Texture& texture)
	{
		// Channel descriptor for 4 Channels (RGBA)
		cudaChannelFormatDesc channelDesc;
		if (texture.HDR)
			channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		else
			channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaArray_t cuArray;

		CheckCudaErrors(cudaMallocArray(&cuArray, &channelDesc, texture.width, texture.height));

		uint32_t elementSize = 4 * (texture.HDR ? sizeof(float) : sizeof(unsigned char));
		const size_t spitch = texture.width * elementSize;
		CheckCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, texture.pixels, spitch, texture.width * elementSize, texture.height, cudaMemcpyHostToDevice));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.sRGB = texture.sRGB;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = texture.HDR ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		cudaTextureObject_t texObject = 0;
		CheckCudaErrors(cudaCreateTextureObject(&texObject, &resDesc, &texDesc, NULL));

		return texObject;
	}

	void Texture::DestructFromDevice(const cudaTextureObject_t& texture)
	{
		cudaResourceDesc resDesc;
		CheckCudaErrors(cudaGetTextureObjectResourceDesc(&resDesc, texture));
		CheckCudaErrors(cudaDestroyTextureObject(texture));
		CheckCudaErrors(cudaFreeArray(resDesc.res.array.array));
	}

	void Texture::CopyFrom(const Texture& other)
	{
		width = other.width;
		height = other.height;
		channels = other.channels;
		sRGB = other.sRGB;
		HDR = other.HDR;
		type = other.type;

		if (HDR)
		{
			pixels = new float[channels * width * height];
			memcpy(pixels, other.pixels, sizeof(float) * channels * width * height);
		}
		else
		{
			pixels = new char[channels * width * height];
			memcpy(pixels, other.pixels, sizeof(char) * channels * width * height);
		}
	}

}
