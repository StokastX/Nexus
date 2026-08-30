#include "GLVertexArray.h"

#include <GL/glew.h>
#include <cassert>
#include <utility>

namespace Nexus {

	static GLenum ShaderDataTypeToGLBaseType(ShaderDataType type)
	{
		switch (type)
		{
			case ShaderDataType::Float:  case ShaderDataType::Float2:
			case ShaderDataType::Float3: case ShaderDataType::Float4:
			case ShaderDataType::Mat3:   case ShaderDataType::Mat4:
				return GL_FLOAT;
			case ShaderDataType::Int:    case ShaderDataType::Int2:
			case ShaderDataType::Int3:   case ShaderDataType::Int4:
				return GL_INT;
			case ShaderDataType::UInt:   case ShaderDataType::UInt2:
			case ShaderDataType::UInt3:  case ShaderDataType::UInt4:
				return GL_UNSIGNED_INT;
			case ShaderDataType::UByte4:
				return GL_UNSIGNED_BYTE;
			case ShaderDataType::None:
				break;
		}

		// Note the &&, not a comma: assert(false, "message") asserts on the string literal,
		// which is a non-null pointer, and therefore never fires.
		assert(false && "Unknown ShaderDataType");
		return GL_FLOAT;
	}

	GLVertexArray::GLVertexArray()
	{
		glCreateVertexArrays(1, m_Handle.AddressOf());
	}

	void GLVertexArray::Bind() const
	{
		glBindVertexArray(m_Handle.Get());
	}

	void GLVertexArray::Unbind() const
	{
		glBindVertexArray(0);
	}

	void GLVertexArray::AddVertexBuffer(GLVertexBuffer&& vertexBuffer)
	{
		const BufferLayout& layout = vertexBuffer.GetLayout();
		assert(!layout.IsEmpty() && "Vertex buffer has no layout");
		assert(layout.GetStride() > 0 && "Vertex buffer layout has zero stride");

		const uint32_t bindingIndex = m_BindingIndex++;
		glVertexArrayVertexBuffer(m_Handle.Get(), bindingIndex, vertexBuffer.GetHandle(), 0, layout.GetStride());

		for (const BufferElement& element : layout)
		{
			const uint32_t componentCount = element.GetComponentCount();
			const GLenum baseType = ShaderDataTypeToGLBaseType(element.Type);

			// Every type but a matrix is a single slot, so this loop runs once for them.
			for (uint32_t slot = 0; slot < element.GetSlotCount(); slot++)
			{
				const uint32_t attributeIndex = m_AttributeIndex++;
				const uint32_t relativeOffset = element.Offset + slot * element.GetSlotSize();

				glEnableVertexArrayAttrib(m_Handle.Get(), attributeIndex);

				if (ShaderDataTypeIsInteger(element.Type))
					glVertexArrayAttribIFormat(m_Handle.Get(), attributeIndex, componentCount, baseType, relativeOffset);
				else
					glVertexArrayAttribFormat(m_Handle.Get(), attributeIndex, componentCount, baseType,
						element.Normalized ? GL_TRUE : GL_FALSE, relativeOffset);

				glVertexArrayAttribBinding(m_Handle.Get(), attributeIndex, bindingIndex);
			}
		}

		m_VertexBuffers.push_back(std::move(vertexBuffer));
	}

	void GLVertexArray::SetIndexBuffer(GLIndexBuffer&& indexBuffer)
	{
		glVertexArrayElementBuffer(m_Handle.Get(), indexBuffer.GetHandle());
		m_IndexBuffer.emplace(std::move(indexBuffer));
	}

}
