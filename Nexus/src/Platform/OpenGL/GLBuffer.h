#pragma once
#include <cstdint>

#include "GLHandle.h"
#include "Resources/BufferLayout.h"

namespace Nexus {

	/*
	 * Concrete OpenGL buffer objects. Deliberately not virtual: the renderer targets one graphics
	 * API, and the CUDA interop some resources exist for (see InteropTexture) cannot be expressed
	 * through a backend-neutral interface anyway.
	 *
	 * Both types are move-only, inherited from the GLHandle members.
	 */

	class GLVertexBuffer
	{
	public:
		// Dynamic buffer of `size` bytes, filled later through SetData.
		explicit GLVertexBuffer(uint32_t size);

		// Static buffer initialised from `vertices`. Takes const void* rather than float* so that
		// any vertex struct can be uploaded directly, without casting at the call site.
		GLVertexBuffer(const void* vertices, uint32_t size);

		void Bind() const;
		void Unbind() const;

		void SetData(const void* data, uint32_t size);

		uint32_t GetHandle() const { return m_Handle.Get(); }
		uint32_t GetSize() const { return m_Size; }

		const BufferLayout& GetLayout() const { return m_Layout; }
		void SetLayout(const BufferLayout& layout) { m_Layout = layout; }

	private:
		GLBufferHandle m_Handle;
		BufferLayout m_Layout;
		uint32_t m_Size = 0;
	};

	class GLIndexBuffer
	{
	public:
		GLIndexBuffer(const uint32_t* indices, uint32_t count);

		void Bind() const;
		void Unbind() const;

		uint32_t GetHandle() const { return m_Handle.Get(); }
		uint32_t GetCount() const { return m_Count; }

	private:
		GLBufferHandle m_Handle;
		uint32_t m_Count = 0;
	};

}
