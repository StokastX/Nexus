#include "GLBuffer.h"

#include <GL/glew.h>
#include <cassert>

namespace Nexus {

	GLVertexBuffer::GLVertexBuffer(uint32_t size)
		: m_Size(size)
	{
		glCreateBuffers(1, m_Handle.AddressOf());
		glNamedBufferData(m_Handle.Get(), size, nullptr, GL_DYNAMIC_DRAW);
	}

	GLVertexBuffer::GLVertexBuffer(const void* vertices, uint32_t size)
		: m_Size(size)
	{
		glCreateBuffers(1, m_Handle.AddressOf());
		glNamedBufferData(m_Handle.Get(), size, vertices, GL_STATIC_DRAW);
	}

	void GLVertexBuffer::Bind() const
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_Handle.Get());
	}

	void GLVertexBuffer::Unbind() const
	{
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void GLVertexBuffer::SetData(const void* data, uint32_t size)
	{
		assert(size <= m_Size && "SetData would write past the end of the buffer");
		glNamedBufferSubData(m_Handle.Get(), 0, size, data);
	}

	GLIndexBuffer::GLIndexBuffer(const uint32_t* indices, uint32_t count)
		: m_Count(count)
	{
		// Direct state access sidesteps the usual bootstrapping problem here: GL_ELEMENT_ARRAY_BUFFER
		// cannot be bound without an active VAO, so the pre-DSA idiom was to upload through
		// GL_ARRAY_BUFFER instead. glNamedBufferData needs no binding at all.
		glCreateBuffers(1, m_Handle.AddressOf());
		glNamedBufferData(m_Handle.Get(), static_cast<GLsizeiptr>(count) * sizeof(uint32_t), indices, GL_STATIC_DRAW);
	}

	void GLIndexBuffer::Bind() const
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Handle.Get());
	}

	void GLIndexBuffer::Unbind() const
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

}
