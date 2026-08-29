#pragma once
#include "Resources/VertexBuffer.h"

namespace Nexus {

	class GLVertexBuffer : public VertexBuffer
	{
	public:
		GLVertexBuffer(uint32_t size);
		GLVertexBuffer(float* vertices, uint32_t size);
		virtual ~GLVertexBuffer();

		virtual void Bind() const override;
		virtual void Unbind() const override;
		
		virtual void SetData(const void* data, uint32_t size);

		virtual const BufferLayout& GetLayout() const override { return m_Layout; }
		virtual void SetLayout(const BufferLayout& layout) override { m_Layout = layout; }

	private:
		BufferLayout m_Layout;
		uint32_t m_Handle = 0;
	};

	class GLIndexBuffer : public IndexBuffer
	{
	public:
		GLIndexBuffer(uint32_t* indices, uint32_t count);
		virtual ~GLIndexBuffer();

		virtual void Bind() const;
		virtual void Unbind() const;

		virtual uint32_t GetCount() const { return m_Count; }
	private:
		uint32_t m_Count;
		uint32_t m_Handle = 0;
	};

}

