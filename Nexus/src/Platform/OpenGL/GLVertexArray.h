#pragma once
#include <cstdint>
#include <optional>
#include <vector>

#include "GLBuffer.h"
#include "GLHandle.h"

namespace Nexus {

	class GLVertexArray
	{
	public:
		GLVertexArray();

		void Bind() const;
		void Unbind() const;

		/*
		 * Takes ownership of the buffer. A vertex array references its vertex buffers by name, so
		 * they have to outlive it; storing them here makes that automatic instead of a rule to
		 * remember. Buffers are stored by value rather than behind a shared_ptr because nothing in
		 * this renderer shares one buffer between two vertex arrays.
		 *
		 * Attribute locations are handed out sequentially across calls, so the first buffer's
		 * attributes start at location 0, the next buffer's continue from there.
		 */
		void AddVertexBuffer(GLVertexBuffer&& vertexBuffer);
		void SetIndexBuffer(GLIndexBuffer&& indexBuffer);

		uint32_t GetHandle() const { return m_Handle.Get(); }

		GLVertexBuffer& GetVertexBuffer(size_t index) { return m_VertexBuffers[index]; }
		const GLVertexBuffer& GetVertexBuffer(size_t index) const { return m_VertexBuffers[index]; }
		const std::vector<GLVertexBuffer>& GetVertexBuffers() const { return m_VertexBuffers; }

		bool HasIndexBuffer() const { return m_IndexBuffer.has_value(); }
		const GLIndexBuffer& GetIndexBuffer() const { return *m_IndexBuffer; }

	private:
		GLVertexArrayHandle m_Handle;

		// Next free shader attribute location, and next free vertex buffer binding point. Under
		// the separate format/binding model these are distinct namespaces: one buffer occupies a
		// single binding point but may feed several attribute locations.
		uint32_t m_AttributeIndex = 0;
		uint32_t m_BindingIndex = 0;

		std::vector<GLVertexBuffer> m_VertexBuffers;
		std::optional<GLIndexBuffer> m_IndexBuffer;
	};

}
