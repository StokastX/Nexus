#include "PixelBuffer.h"
#include "../Utils/Utils.h"

PixelBuffer::PixelBuffer(uint2 resolution)
    : m_Resolution(resolution)
{
    glGenBuffers(1, &m_Handle);
    Bind();
    glBufferData(GL_PIXEL_UNPACK_BUFFER, resolution.x * resolution.y * sizeof(uint32_t), NULL, GL_DYNAMIC_DRAW);

    // Register the buffer for CUDA to use
	CheckCudaErrors(cudaGraphicsGLRegisterBuffer(&m_CudaResource, m_Handle, cudaGraphicsRegisterFlagsWriteDiscard));
    Unbind();
}

void PixelBuffer::Bind() const
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_Handle);
}

void PixelBuffer::Unbind() const
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void PixelBuffer::OnResize(uint2 resolution)
{
    m_Resolution = resolution;
    Bind();
    CheckCudaErrors(cudaGraphicsUnregisterResource(m_CudaResource));
    glBufferData(GL_PIXEL_UNPACK_BUFFER, resolution.x * resolution.y * sizeof(uint32_t), NULL, GL_DYNAMIC_DRAW);
	CheckCudaErrors(cudaGraphicsGLRegisterBuffer(&m_CudaResource, m_Handle, cudaGraphicsRegisterFlagsWriteDiscard));
    Unbind();
}

PixelBuffer::~PixelBuffer()
{
    CheckCudaErrors(cudaGraphicsUnregisterResource(m_CudaResource));
    glDeleteBuffers(1, &m_Handle);
}
