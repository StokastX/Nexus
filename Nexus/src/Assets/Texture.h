#pragma once
#include <iostream>
#include <cstdint>
#include <memory>
#include <cuda_runtime_api.h>
#include <Utils/cuda_math.h>
#include "Device/CudaTexture.h"

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
			NORMALS
		};

		Texture() = default;

		// Takes ownership of the buffer. Passing a StbImageData rather than a raw void* is what
		// makes that visible at the call site.
		Texture(uint32_t w, uint32_t h, uint32_t c, bool isHDR, StbImageData d);

		// Nothing in the codebase copies a Texture -- they are held through shared_ptr, and the
		// device-resident copy is the CudaTexture member below.
		Texture(const Texture&) = delete;
		Texture& operator=(const Texture&) = delete;

		// Spelled out because declaring the copy operations above -- even as deleted -- suppresses
		// the implicitly generated move ones.
		Texture(Texture&&) noexcept = default;
		Texture& operator=(Texture&&) noexcept = default;

		// Uploads the pixels and builds the sampler. Explicit rather than done in the constructor,
		// because sRGB is set after loading (Scene::AddHDRMap clears it) and because IMGLoader
		// produces Textures that never reach the GPU at all.
		void UploadToDevice();

		uint32_t width = 0;
		uint32_t height = 0;

		// What the source file held. The buffer is always 4 components wide regardless, because
		// IMGLoader asks stb for 4 -- so never size an allocation or a copy from this.
		uint32_t channels = 0;

		bool sRGB = true;
		bool HDR = false;

		StbImageData pixels;

		// The device-resident copy, owned here the same way Mesh owns its DeviceVectors.
		CudaTexture deviceTexture;
		Type type = Type::DIFFUSE;
	};

}
