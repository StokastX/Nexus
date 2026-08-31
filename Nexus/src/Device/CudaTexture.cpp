#include "CudaTexture.h"

#include <cstring>

#include "Utils/Utils.h"

namespace Nexus {

	CudaTexture::CudaTexture(const void* pixels, uint32_t width, uint32_t height, bool hdr, bool sRGB)
	{
		cudaChannelFormatDesc channelDesc;
		if (hdr)
			channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		else
			channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

		CheckCudaErrors(cudaMallocArray(&m_Array, &channelDesc, width, height));

		const uint32_t elementSize = 4 * (hdr ? sizeof(float) : sizeof(unsigned char));
		const size_t pitch = width * elementSize;
		CheckCudaErrors(cudaMemcpy2DToArray(m_Array, 0, 0, pixels, pitch, pitch, height, cudaMemcpyHostToDevice));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = m_Array;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.sRGB = sRGB;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = hdr ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		CheckCudaErrors(cudaCreateTextureObject(&m_Object, &resDesc, &texDesc, NULL));
	}

	CudaTexture::~CudaTexture()
	{
		Destroy();
	}

	CudaTexture::CudaTexture(CudaTexture&& other) noexcept
		: m_Array(other.m_Array), m_Object(other.m_Object)
	{
		other.m_Array = nullptr;
		other.m_Object = 0;
	}

	CudaTexture& CudaTexture::operator=(CudaTexture&& other) noexcept
	{
		if (this != &other)
		{
			Destroy();
			m_Array = other.m_Array;
			m_Object = other.m_Object;
			other.m_Array = nullptr;
			other.m_Object = 0;
		}
		return *this;
	}

	/*
	 * Descriptor first, then the memory it describes. Keeping the array as a member is what makes
	 * that possible -- the previous DestructFromDevice had to recover it from the object with
	 * cudaGetTextureObjectResourceDesc before it could free anything.
	 *
	 * Like cudaDestroySurfaceObject, cudaDestroyTextureObject is not stream-ordered, so this must
	 * not run while a kernel is still sampling the texture. Every caller tears textures down
	 * between frames (a scene reset, or replacing the HDR map), never mid-pass.
	 */
	void CudaTexture::Destroy()
	{
		if (m_Object)
			CheckCudaErrors(cudaDestroyTextureObject(m_Object));

		if (m_Array)
			CheckCudaErrors(cudaFreeArray(m_Array));

		m_Object = 0;
		m_Array = nullptr;
	}

}
