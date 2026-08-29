#pragma once
#include <iostream>
#include <type_traits>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "Device/CudaGraphicsResource.h"
#include "Platform/OpenGL/GLHandle.h"

namespace Nexus {

	class PixelBuffer
	{
	public:
		PixelBuffer(uint2 resolution);

		void Bind() const;
		void Unbind() const;
		void OnResize(uint2 resolution);

		uint32_t GetWidth() const { return m_Resolution.x; };
		uint32_t GetHeight() const { return m_Resolution.y; };
		unsigned int GetHandle() const { return m_Handle.Get(); };
		cudaGraphicsResource_t& GetCudaResource() { return m_CudaResource.Get(); };

	private:
		GLBufferHandle m_Handle;
		CudaGraphicsResource m_CudaResource;
		uint2 m_Resolution;
	};

	// PixelBuffer owns two independent lifetimes, a GL buffer name and a CUDA registration.
	// Copying one used to duplicate both; the second unregister is a CUDA error that takes the
	// process down through CheckCudaErrors, far from the copy that caused it.
	static_assert(!std::is_copy_constructible_v<PixelBuffer>, "PixelBuffer must stay move-only");
	static_assert(!std::is_copy_assignable_v<PixelBuffer>, "PixelBuffer must stay move-only");

}
