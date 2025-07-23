#pragma once

#include "Utils/Utils.h"
#include <cuda_runtime_api.h>
#include <cstdint>


class CudaMemory
{
public:
	template<typename T>
	static T* Allocate(uint32_t count)
	{
		T* ptr;
		CheckCudaErrors(cudaMalloc((void**)&ptr, sizeof(T) * count));
		return ptr;
	}

	template<typename T>
	static T* AllocateAsync(uint32_t count)
	{
		T* ptr;
		CheckCudaErrors(cudaMallocAsync((void**)&ptr, sizeof(T) * count, 0));
		return ptr;
	}

	template<typename THost, typename TDevice = THost>
	static TDevice* AllocBuffer(THost* hostBuffer, uint32_t count)
	{
		TDevice* deviceBuffer = Allocate<TDevice>(count);
		if constexpr (!is_trivially_copyable_to_device<THost>)
		{
			std::vector<TDevice> temp(count);
			for (size_t i = 0; i < count; i++)
				temp[i] = THost::ToDevice(hostBuffer[i]);
			
			Copy<TDevice>(deviceBuffer, temp, count, cudaMemcpyHostToDevice);
		}
		else
			Copy<TDevice>(deviceBuffer, hostBuffer, count, cudaMemcpyHostToDevice);
	}

	template<typename T>
	static void Copy(T* dst, T* src, uint32_t count, cudaMemcpyKind kind)
	{
		CheckCudaErrors(cudaMemcpy((void*)dst, (void*)src, sizeof(T) * count, kind));
	}

	template<typename T>
	static void CopyAsync(T* dst, T* src, uint32_t count, cudaMemcpyKind kind)
	{
		CheckCudaErrors(cudaMemcpyAsync((void*)dst, (void*)src, sizeof(T) * count, kind));
	}

	static void MemsetAsync(void* dst, uint32_t value, uint32_t count)
	{
		CheckCudaErrors(cudaMemsetAsync(dst, value, count));
	}

	static void Free(void* ptr)
	{
		CheckCudaErrors(cudaFree(ptr));
	}

	static void FreeAsync(void* ptr)
	{
		CheckCudaErrors(cudaFreeAsync(ptr, 0));
	}
};