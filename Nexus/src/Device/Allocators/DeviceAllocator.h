#pragma once
#include <cuda_runtime_api.h>
#include "Device/CudaMemory.h"

namespace Nexus {

	template<typename T>
	class DeviceAllocator
	{
	public:
		DeviceAllocator() = default;

		static T* Alloc(DeviceAllocator* allocator, size_t count)
		{
			if (!allocator)
				return CudaMemory::Allocate<T>(count);
			else
				return allocator->Alloc(count);
		}

		static void Free(DeviceAllocator* allocator, T* ptr)
		{
			if (!allocator)
				CudaMemory::Free(ptr);
			else
				allocator->Free(ptr);
		}


	protected:
		virtual T* Alloc(size_t count)
		{
			return CudaMemory::Allocate<T>(count);
		}

		virtual void Free(T* ptr)
		{
			CudaMemory::Free(ptr);
		}
	};

}