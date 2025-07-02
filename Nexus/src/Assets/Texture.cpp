#include "Texture.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include "Utils/Utils.h"


Texture::Texture(uint32_t w, uint32_t h, uint32_t c, unsigned char* d) : width(w), height(h), channels(c), pixels(d)
{
}

cudaTextureObject_t Texture::ToDevice(const Texture& texture)
{
	// Channel descriptor for 4 Channels (RGBA)
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaArray_t cuArray;

	CheckCudaErrors(cudaMallocArray(&cuArray, &channelDesc, texture.width, texture.height));

	const size_t spitch = texture.width * 4 * sizeof(unsigned char);
	CheckCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, texture.pixels, spitch, texture.width * 4 * sizeof(unsigned char), texture.height, cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.sRGB = 1;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
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

