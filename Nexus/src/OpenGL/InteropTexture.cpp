#include "InteropTexture.h"

#include <cstring>

// glew must precede cuda_gl_interop.h -- see the note in CudaGraphicsResource.cpp.
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "Utils/Utils.h"

namespace Nexus {

	InteropTexture::InteropTexture(uint2 resolution)
		: m_Texture(resolution)
	{
		Register();
	}

	InteropTexture::~InteropTexture()
	{
		DestroySurface();
	}

	void InteropTexture::Register()
	{
		// SurfaceLoadStore is what makes the mapped array usable as a surface rather than only as a
		// copy target. WriteDiscard is honest here: the accumulate kernel writes every pixel of the
		// texture every frame, so CUDA never needs to see the previous contents.
		m_Resource.RegisterImage(m_Texture.GetHandle(), GL_TEXTURE_2D,
			cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard);
	}

	void InteropTexture::OnResize(uint2 resolution)
	{
		// The surface object goes first: it describes the array behind the current registration,
		// and both that array and the texture storage under it are about to stop existing.
		DestroySurface();

		// OGLTexture::OnResize reallocates the texture's storage through glTexImage2D, which leaves
		// the registration pointing at storage that no longer exists. It has to be dropped first and
		// rebuilt against the new allocation afterwards.
		m_Resource.Unregister();
		m_Texture.OnResize(resolution);
		Register();
	}

	cudaSurfaceObject_t InteropTexture::MapSurface()
	{
		m_Resource.Map();

		// The array has to be re-fetched on every mapping -- CUDA does not promise that a mapping
		// hands back the array the previous one did. What can be reused is the surface object built
		// on it, as long as the array really is the same one.
		//
		// Reusing it is not a micro-optimisation. cudaDestroySurfaceObject takes no stream and is
		// not stream-ordered: it runs on the host the moment it is called, while the accumulate
		// kernel that writes through the surface is still in flight. Destroying it once per frame
		// would mean tearing down a descriptor a running kernel is using, and the only way to make
		// that safe per frame is a full synchronisation in the middle of every frame.
		cudaArray_t mappedArray = m_Resource.GetMappedArray();

		if (mappedArray != m_MappedArray)
		{
			DestroySurface();

			cudaResourceDesc resourceDesc;
			memset(&resourceDesc, 0, sizeof(resourceDesc));
			resourceDesc.resType = cudaResourceTypeArray;
			resourceDesc.res.array.array = mappedArray;

			CheckCudaErrors(cudaCreateSurfaceObject(&m_Surface, &resourceDesc));
			m_MappedArray = mappedArray;
		}

		return m_Surface;
	}

	void InteropTexture::UnmapSurface()
	{
		// The surface object deliberately outlives the mapping. It is only ever dereferenced by
		// kernels launched between Map and Unmap, and MapSurface revalidates it against the array
		// before handing it out again.
		m_Resource.Unmap();
	}

	void InteropTexture::DestroySurface()
	{
		if (!m_Surface)
			return;

		// Kernels referencing the surface object may still be running, and destroying it is not
		// stream-ordered, so the wait has to be explicit. This only runs on resize, on teardown, or
		// on the rare mapping that returns a different array -- never in the steady-state frame.
		CheckCudaErrors(cudaDeviceSynchronize());

		CheckCudaErrors(cudaDestroySurfaceObject(m_Surface));
		m_Surface = 0;
		m_MappedArray = nullptr;
	}

}
