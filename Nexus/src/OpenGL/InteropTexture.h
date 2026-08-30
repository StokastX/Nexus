#pragma once
#include <type_traits>
#include <cuda_runtime_api.h>

#include "OGLTexture.h"
#include "Device/CudaGraphicsResource.h"

namespace Nexus {

	/*
	 * An OGLTexture that CUDA writes into directly, through a surface object.
	 *
	 * Only the path tracer's output is registered this way. OGLRenderer's colour and instance-id
	 * attachments stay plain OGLTextures -- they are written by the rasteriser, and registering
	 * them would constrain the driver for nothing.
	 */
	class InteropTexture
	{
	public:
		InteropTexture(uint2 resolution);
		~InteropTexture();

		void OnResize(uint2 resolution);

		// Hands the texture to CUDA. The returned surface object may only be used while the texture
		// is mapped, so it must not outlive the matching UnmapSurface at any call site.
		cudaSurfaceObject_t MapSurface();
		void UnmapSurface();

		// Mirrors of the OGLTexture accessors, so that call sites drawing the render image do not
		// care which of the two they were handed.
		void Bind() { m_Texture.Bind(); }
		unsigned int GetHandle() const { return m_Texture.GetHandle(); }
		uint32_t GetWidth() { return m_Texture.GetWidth(); }
		uint32_t GetHeight() { return m_Texture.GetHeight(); }
		uint2 GetResolution() { return m_Texture.GetResolution(); }

	private:
		void Register();

		// Tears down the surface object, waiting for in-flight kernels first. See the note in the
		// .cpp -- this is the whole reason the object is cached rather than rebuilt per frame.
		void DestroySurface();

		// Declaration order is the destruction order reversed: the registration is torn down before
		// the GL name it refers to.
		OGLTexture m_Texture;
		CudaGraphicsResource m_Resource;

		// The array the current surface object was built on, kept so that a mapping handing back
		// the same array can reuse it. Reset to null whenever the surface object goes away, so that
		// a recycled array address can never be mistaken for the one still described.
		cudaArray_t m_MappedArray = nullptr;
		cudaSurfaceObject_t m_Surface = 0;
	};

	// An InteropTexture owns a GL texture name and a CUDA registration, and a copy would hand two
	// owners the same pair. Both members already delete their copy operations, so these only assert
	// that no future member quietly restores them.
	static_assert(!std::is_copy_constructible_v<InteropTexture>, "InteropTexture must stay move-only");
	static_assert(!std::is_copy_assignable_v<InteropTexture>, "InteropTexture must stay move-only");

}
