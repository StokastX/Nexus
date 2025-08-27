#pragma once
#include <iostream>
#include <cuda_runtime.h>

class OGLTexture
{
public:
	OGLTexture(uint2 resolution);
	~OGLTexture();

	void Bind();
	void OnResize(uint2 resolution);

	unsigned int GetHandle() { return m_Handle; };
	uint32_t GetWidth() { return m_Resolution.x; };
	uint32_t GetHeight() { return m_Resolution.y; };
	uint2 GetResolution() { return m_Resolution; };

private:
	unsigned int m_Handle = 0;
	uint2 m_Resolution;
};