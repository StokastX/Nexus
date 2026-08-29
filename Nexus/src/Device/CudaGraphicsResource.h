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
 * Wrapping it lets the enclosing class (PixelBuffer) declare no special members at all.
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
			: m_Resource(other.m_Resource)
		{
			other.m_Resource = nullptr;
		}

		CudaGraphicsResource& operator=(CudaGraphicsResource&& other) noexcept
		{
			if (this != &other)
			{
				Unregister();
				m_Resource = other.m_Resource;
				other.m_Resource = nullptr;
			}
			return *this;
		}

		// Registers an OpenGL buffer for CUDA access, releasing any previous registration.
		void RegisterBuffer(uint32_t glBuffer, unsigned int flags);

		// Releases the registration. Safe to call on an empty resource.
		void Unregister();

		// Non-const overload returns a reference because the CUDA map/unmap entry points take
		// the address of the resource.
		cudaGraphicsResource_t& Get() { return m_Resource; }
		cudaGraphicsResource_t Get() const { return m_Resource; }

		explicit operator bool() const { return m_Resource != nullptr; }

	private:
		cudaGraphicsResource_t m_Resource = nullptr;
	};

}
