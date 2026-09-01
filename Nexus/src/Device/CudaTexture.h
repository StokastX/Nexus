#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

/*
 * Move-only owner of a CUDA texture: the cudaArray_t holding the pixels, and the
 * cudaTextureObject_t describing how they are sampled.
 *
 * The two are a single lifetime -- the object is only a descriptor over the array, so releasing
 * either one alone leaves the other dangling.
 *
 * Takes a raw pixel pointer rather than an Assets::Texture on purpose: Device/ must not depend on
 * Assets/, and the HDR map or any future image source can then reuse the same wrapper.
 */
namespace Nexus {

	class CudaTexture
	{
	public:
		CudaTexture() = default;

		/*
		 * Uploads `pixels` and builds a sampler over it: wrapped, linearly filtered, normalized
		 * coordinates. The buffer is always four components wide -- one float each when `hdr`,
		 * one unsigned char each otherwise -- because TextureLoader always asks stb for 4, so never
		 * size this from a texture's `channels`. `sRGB` asks the hardware for the sRGB -> linear
		 * conversion on fetch.
		 */
		CudaTexture(const void* pixels, uint32_t width, uint32_t height, bool hdr, bool sRGB);

		~CudaTexture();

		CudaTexture(const CudaTexture&) = delete;
		CudaTexture& operator=(const CudaTexture&) = delete;

		CudaTexture(CudaTexture&& other) noexcept;
		CudaTexture& operator=(CudaTexture&& other) noexcept;

		// Zero when empty, which is the value D_Scene already reads as "no texture".
		cudaTextureObject_t Handle() const { return m_Object; }

		explicit operator bool() const { return m_Object != 0; }

	private:
		void Destroy();

		cudaArray_t m_Array = nullptr;
		cudaTextureObject_t m_Object = 0;
	};

}
