#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "Platform/OpenGL/GLHandle.h"

namespace Nexus {

	class OGLTexture
	{
	public:
		OGLTexture(uint2 resolution);

		void Bind();
		void OnResize(uint2 resolution);

		unsigned int GetHandle() const { return m_Handle.Get(); };
		uint32_t GetWidth() { return m_Resolution.x; };
		uint32_t GetHeight() { return m_Resolution.y; };
		uint2 GetResolution() { return m_Resolution; };

	private:
		GLTextureHandle m_Handle;
		uint2 m_Resolution;
	};

}