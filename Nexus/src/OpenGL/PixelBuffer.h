#pragma once
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

class PixelBuffer
{
public:
	PixelBuffer(uint2 resolution);
	~PixelBuffer();

	void Bind() const;
	void Unbind() const;
	void OnResize(uint2 resolution);

	uint32_t GetWidth() const { return m_Resolution.x; };
	uint32_t GetHeight() const { return m_Resolution.y; };
	unsigned int GetHandle() { return m_Handle; };
	cudaGraphicsResource_t& GetCudaResource() { return m_CudaResource; };

private:
	unsigned int m_Handle = 0;
	cudaGraphicsResource_t m_CudaResource;
	uint2 m_Resolution;
};
