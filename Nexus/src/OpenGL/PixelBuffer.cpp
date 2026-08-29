#include "PixelBuffer.h"
#include "../Utils/Utils.h"

namespace Nexus {

    PixelBuffer::PixelBuffer(uint2 resolution)
        : m_Resolution(resolution)
    {
        glGenBuffers(1, m_Handle.AddressOf());
        Bind();
        glBufferData(GL_PIXEL_UNPACK_BUFFER, resolution.x * resolution.y * sizeof(uint32_t), NULL, GL_DYNAMIC_DRAW);

        // Register the buffer for CUDA to use
        m_CudaResource.RegisterBuffer(m_Handle.Get(), cudaGraphicsRegisterFlagsWriteDiscard);
        Unbind();
    }

    void PixelBuffer::Bind() const
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_Handle.Get());
    }

    void PixelBuffer::Unbind() const
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    void PixelBuffer::OnResize(uint2 resolution)
    {
        m_Resolution = resolution;
        Bind();
        m_CudaResource.Unregister();
        glBufferData(GL_PIXEL_UNPACK_BUFFER, resolution.x * resolution.y * sizeof(uint32_t), NULL, GL_DYNAMIC_DRAW);
        m_CudaResource.RegisterBuffer(m_Handle.Get(), cudaGraphicsRegisterFlagsWriteDiscard);
        Unbind();
    }

}
