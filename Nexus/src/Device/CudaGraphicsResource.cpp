#include "CudaGraphicsResource.h"

// glew must precede cuda_gl_interop.h: the latter pulls in the Windows SDK's <GL/gl.h>,
// which does not compile on its own because WINGDIAPI and APIENTRY come from <windows.h>.
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "Utils/Utils.h"

namespace Nexus {

	void CudaGraphicsResource::RegisterBuffer(uint32_t glBuffer, unsigned int flags)
	{
		Unregister();
		CheckCudaErrors(cudaGraphicsGLRegisterBuffer(&m_Resource, glBuffer, flags));
	}

	void CudaGraphicsResource::Unregister()
	{
		if (!m_Resource)
			return;

		// CheckCudaErrors prints and exits rather than throwing, so this is safe to call from a
		// destructor. Behaviour is unchanged from the hand-written ~PixelBuffer this replaces.
		CheckCudaErrors(cudaGraphicsUnregisterResource(m_Resource));
		m_Resource = nullptr;
	}

}
