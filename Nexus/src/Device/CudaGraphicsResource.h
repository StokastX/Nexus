#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

/*
 * Move-only owner of a CUDA graphics interop registration.
 *
 * A registered resource is a second, independent lifetime living alongside the OpenGL object it
 * wraps: the GL name is freed with glDelete*, the registration with cudaGraphicsUnregisterResource.
 * Holding it as a bare cudaGraphicsResource_t has the same failure mode as a bare GL name -- an
 * implicitly generated copy hands two owners the same registration, and the second unregister is
 * a CUDA error rather than a crash, so it goes unnoticed.
 *
 * Wrapping it lets the enclosing class (InteropTexture) declare no special members at all.
 */
namespace Nexus {

	class CudaGraphicsResource
	{
	public:
		CudaGraphicsResource() = default;
		~CudaGraphicsResource() { Unregister(); }

		CudaGraphicsResource(const CudaGraphicsResource&) = delete;
		CudaGraphicsResource& operator=(const CudaGraphicsResource&) = delete;

		CudaGraphicsResource(CudaGraphicsResource&& other) noexcept
			: m_Resource(other.m_Resource), m_Mapped(other.m_Mapped)
		{
			other.m_Resource = nullptr;
			other.m_Mapped = false;
		}

		CudaGraphicsResource& operator=(CudaGraphicsResource&& other) noexcept
		{
			if (this != &other)
			{
				Unregister();
				m_Resource = other.m_Resource;
				m_Mapped = other.m_Mapped;
				other.m_Resource = nullptr;
				other.m_Mapped = false;
			}
			return *this;
		}

		// Registers an OpenGL texture for CUDA access, releasing any previous registration.
		// `target` is a GLenum, taken as unsigned int so that including this header does not drag
		// in glew -- same reason the GLDelete adapters are declared over uint32_t.
		void RegisterImage(uint32_t glTexture, unsigned int target, unsigned int flags);

		// Releases the registration. Safe to call on an empty resource.
		void Unregister();

		// Hand the resource to CUDA and back. OpenGL must not touch the underlying object while it
		// is mapped. Both are idempotent, so the mapped state can never go out of step with the
		// number of calls made.
		void Map();
		void Unmap();

		// The array backing an image registration. Only meaningful between Map and Unmap, and the
		// array a mapping returns is not guaranteed to be the one the previous mapping returned --
		// re-fetch it after every Map rather than caching it across frames.
		cudaArray_t GetMappedArray(uint32_t arrayIndex = 0, uint32_t mipLevel = 0) const;

		bool IsMapped() const { return m_Mapped; }

		// Non-const overload returns a reference because the CUDA map/unmap entry points take
		// the address of the resource.
		cudaGraphicsResource_t& Get() { return m_Resource; }
		cudaGraphicsResource_t Get() const { return m_Resource; }

		explicit operator bool() const { return m_Resource != nullptr; }

	private:
		cudaGraphicsResource_t m_Resource = nullptr;
		bool m_Mapped = false;
	};

}
