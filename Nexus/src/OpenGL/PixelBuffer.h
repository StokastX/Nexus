#pragma once
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

class PixelBuffer
{
public:
	PixelBuffer(uint32_t width, uint32_t height);
	~PixelBuffer();

	void Bind() const;
	void Unbind() const;
	void OnResize(uint32_t width, uint32_t height);

	uint32_t GetWidth() const { return m_Width; };
	uint32_t GetHeight() const { return m_Height; };
	unsigned int GetHandle() { return m_Handle; };
	cudaGraphicsResource_t& GetCudaResource() { return m_CudaResource; };

private:
	unsigned int m_Handle = 0;
	cudaGraphicsResource_t m_CudaResource;
	uint32_t m_Width, m_Height;
};
