#include "CudaGraphicsResource.h"

// glew must precede cuda_gl_interop.h: the latter pulls in the Windows SDK's <GL/gl.h>,
// which does not compile on its own because WINGDIAPI and APIENTRY come from <windows.h>.
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "Utils/Utils.h"

namespace Nexus {

	void CudaGraphicsResource::RegisterImage(uint32_t glTexture, unsigned int target, unsigned int flags)
	{
		Unregister();
		CheckCudaErrors(cudaGraphicsGLRegisterImage(&m_Resource, glTexture, target, flags));
	}

	void CudaGraphicsResource::Unregister()
	{
		if (!m_Resource)
			return;

		// Unregistering a resource that is still mapped is a CUDA error.
		Unmap();

		// CheckCudaErrors prints and exits rather than throwing, so this is safe to call from a
		// destructor.
		CheckCudaErrors(cudaGraphicsUnregisterResource(m_Resource));
		m_Resource = nullptr;
	}

	void CudaGraphicsResource::Map()
	{
		if (m_Mapped)
			return;

		CheckCudaErrors(cudaGraphicsMapResources(1, &m_Resource));
		m_Mapped = true;
	}

	void CudaGraphicsResource::Unmap()
	{
		if (!m_Mapped)
			return;

		CheckCudaErrors(cudaGraphicsUnmapResources(1, &m_Resource, 0));
		m_Mapped = false;
	}

	cudaArray_t CudaGraphicsResource::GetMappedArray(uint32_t arrayIndex, uint32_t mipLevel) const
	{
		cudaArray_t mappedArray = nullptr;
		CheckCudaErrors(cudaGraphicsSubResourceGetMappedArray(&mappedArray, m_Resource, arrayIndex, mipLevel));
		return mappedArray;
	}

}
