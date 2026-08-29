#pragma once
#include "VertexBuffer.h"

#include <iostream>

#include "Platform/OpenGL/GLVertexBuffer.h"


namespace Nexus
{

		std::unique_ptr<VertexBuffer> Create(uint32_t size)
		{
			// For now we only use OpenGL API
			return std::make_unique<GLVertexBuffer>(size);
		}

		std::unique_ptr<VertexBuffer> VertexBuffer::Create(float* vertices, uint32_t size)
		{
			return std::make_unique<GLVertexBuffer>(vertices, size);
		}

		std::unique_ptr<IndexBuffer> IndexBuffer::Create(uint32_t* indices, uint32_t count)
		{
			return std::make_unique<GLIndexBuffer>(indices, count);
		}

}