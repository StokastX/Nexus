#pragma once
#include <iostream>
#include <cstdint>
#include <memory>
#include <cuda_runtime_api.h>
#include <Utils/cuda_math.h>

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

		// Nothing in the codebase copies a Texture -- they are held through shared_ptr and reach the
		// device through ToDevice. The hand-written copy that used to live here allocated with new[]
		// against a free() release, and sized itself from `channels`; both were wrong, and neither
		// was ever noticed because the code never ran. Deleting the copy makes it a compile error.
		Texture(const Texture&) = delete;
		Texture& operator=(const Texture&) = delete;

		// Spelled out because declaring the copy operations above -- even as deleted -- suppresses
		// the implicitly generated move ones.
		Texture(Texture&&) noexcept = default;
		Texture& operator=(Texture&&) noexcept = default;

		static cudaTextureObject_t ToDevice(const Texture& texture);
		static void DestructFromDevice(const cudaTextureObject_t& texture);

		uint32_t width = 0;
		uint32_t height = 0;

		// What the source file held. The buffer is always 4 components wide regardless, because
		// IMGLoader asks stb for 4 -- so never size an allocation or a copy from this.
		uint32_t channels = 0;

		bool sRGB = true;
		bool HDR = false;

		StbImageData pixels;
		Type type = Type::DIFFUSE;
	};

}
