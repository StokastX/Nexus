#pragma once
#include <iostream>
#include <cstdint>
#include <memory>
#include <cuda_runtime_api.h>
#include <Utils/cuda_math.h>
#include "Device/CudaTexture.h"
#include "Device/DeviceTraits.h"

namespace Nexus {

	/*
	 * Releases a pixel buffer back to stb_image.
	 *
	 * Every buffer a Texture holds comes from stbi_load/stbi_loadf, so it has to be released with
	 * stbi_image_free -- not delete, not free. Expressing that as a deleter type moves the
	 * requirement into the pointer's type, instead of leaving it as an unwritten agreement between
	 * IMGLoader (which allocates) and Texture (which releases).
	 *
	 * Stateless on purpose: an empty deleter costs nothing, so a StbImageData is exactly the size of
	 * a bare pointer. operator() is defined in the .cpp so this header need not include stb_image.h.
	 */
	struct StbImageDeleter
	{
		void operator()(void* pixels) const;
	};

	using StbImageData = std::unique_ptr<void, StbImageDeleter>;

	struct Texture
	{
		enum struct Type {
			DIFFUSE,
			ROUGHNESS,
			METALNESS,
			METALLICROUGHNESS,
			EMISSIVE,
			NORMALS,
			ENVIRONMENT
		};

		// Takes ownership of the buffer. Passing a StbImageData rather than a raw void* is what
		// makes that visible at the call site.
		Texture(uint32_t w, uint32_t h, uint32_t c, bool isHDR, Type t, StbImageData d);

		// Nothing in the codebase copies a Texture -- they are held through shared_ptr, and the
		// device-resident copy is the CudaTexture member below.
		Texture(const Texture&) = delete;
		Texture& operator=(const Texture&) = delete;

		// Spelled out because declaring the copy operations above -- even as deleted -- suppresses
		// the implicitly generated move ones.
		Texture(Texture&&) noexcept = default;
		Texture& operator=(Texture&&) noexcept = default;

		// Uploads the pixels and builds the sampler. Explicit rather than done in the constructor
		// because a Texture is a loaded image first and a device resource second; IMGLoader itself
		// needs no CUDA context. A Texture that exists always has pixels -- LoadIMG returns nothing
		// on failure -- so this can never be asked to upload nothing.
		void UploadToDevice();

		/*
		 * Whether the pixels carry an sRGB encoding the sampler has to undo.
		 *
		 * Derived rather than stored: it is a consequence of what the texture means, and holding
		 * it as a second field meant every load site had to set two things in agreement -- which
		 * two of the six already got only by falling through to the default.
		 *
		 * HDR wins over the type: .hdr and .exr store linear radiance as floats, so there is no
		 * encoding to undo. An 8-bit environment map is still sRGB, which is why HDR and not the
		 * role is what makes it linear.
		 */
		bool IsSRGB() const
		{
			if (HDR)
				return false;

			return type == Type::DIFFUSE || type == Type::EMISSIVE || type == Type::ENVIRONMENT;
		}

		uint32_t width = 0;
		uint32_t height = 0;

		// What the source file held. The buffer is always 4 components wide regardless, because
		// IMGLoader asks stb for 4 -- so never size an allocation or a copy from this.
		uint32_t channels = 0;

		bool HDR = false;
		Type type = Type::DIFFUSE;

		StbImageData pixels;

		// The device-resident copy, owned here the same way Mesh owns its DeviceVectors.
		CudaTexture deviceTexture;
	};


	/*
	 * A Texture reaches the device as the bare sampler handle the kernels index -- the pixels and
	 * the sampler itself stay owned by the CudaTexture above. The same arrangement as Mesh, whose
	 * device form is three pointers into containers the Mesh owns.
	 *
	 * Moving a Texture (which a std::vector does when it grows) moves the CudaTexture with it and
	 * leaves the handle unchanged, so handles already uploaded stay valid across a reallocation.
	 */
	template<>
	struct DeviceTraits<Texture>
	{
		using DeviceType = cudaTextureObject_t;

		static cudaTextureObject_t ToDevice(const Texture& texture)
		{
			return texture.deviceTexture.Handle();
		}
	};

}
