#pragma once
#include <vector>
#include <cassert>
#include "Allocators/DeviceAllocator.h"
#include "CudaMemory.h"
#include "DeviceInstance.h"
#include "Memory/Vector.h"

/*
 * Device vector for trivial types (does not handle copy / move constructors
 * since it is not possible to construct an object on the GPU from the host with CUDA).
 * The objects will just be copied over to the device, non trivial conversions
 * from a CPU to GPU instance must therefore be handled by the user.
 * 
 * If THost implements a static TDevice ToDevice() method, this method 
 * will be used to create a device instance and copy it to the device
 * when using the assignment operator.
 * Likewise, if THost implements a static void DestructFromDevice() method,
 * it will be used for destructing the device instance.
 */
template<typename THost, typename TDevice = THost>
class DeviceVector
{
public:
	DeviceVector()
	{
		Realloc(2);
	}

	DeviceVector(size_t size, DeviceAllocator<TDevice>* allocator = nullptr)
		: m_Allocator(allocator)
	{
		Realloc(size);
		m_Size = size;
	}

	DeviceVector(const DeviceVector<THost, TDevice>& other)
		: m_Allocator(other.m_Allocator)
	{
		Realloc(other.Size());
		m_Size = other.Size();
		CudaMemory::CopyAsync<TDevice>(m_Data, other.Data(), other.Size(), cudaMemcpyDeviceToDevice);
	}

	DeviceVector(DeviceVector<THost, TDevice>&& other)
		: m_Allocator(other.m_Allocator), m_Capacity(other.m_Capacity), m_Size(other.m_Size), m_Data(other.m_Data)
	{
		other.m_Data = nullptr;
	}

	DeviceVector(const std::vector<THost>& hostVector, DeviceAllocator<TDevice>* allocator = nullptr)
		: m_Allocator(allocator)
	{
		Realloc(hostVector.size());
		m_Size = hostVector.size();

		if constexpr (is_trivially_copyable_to_device<THost>)
			CudaMemory::CopyAsync<TDevice>(m_Data, (TDevice*)hostVector.data(), hostVector.size(), cudaMemcpyHostToDevice);
		else
		{
			Vector<TDevice> deviceInstances(hostVector.size());
			for (size_t i = 0; i < hostVector.size(); i++)
				deviceInstances[i] = THost::ToDevice(hostVector[i]);

			CudaMemory::CopyAsync<TDevice>(m_Data, deviceInstances.Data(), hostVector.size(), cudaMemcpyHostToDevice);
		}
	}

	DeviceVector(const Vector<THost>& hostVector, DeviceAllocator<TDevice>* allocator = nullptr)
		: m_Allocator(allocator)
	{
		Realloc(hostVector.Size());
		m_Size = hostVector.Size();

		if constexpr (is_trivially_copyable_to_device<THost>)
			CudaMemory::Copy<TDevice>(m_Data, (TDevice*)hostVector.Data(), hostVector.Size(), cudaMemcpyHostToDevice);
		else
		{
			Vector<TDevice> deviceInstances(hostVector.Size());
			for (size_t i = 0; i < hostVector.Size(); i++)
				deviceInstances[i] = THost::ToDevice(hostVector[i]);

			CudaMemory::Copy<TDevice>(m_Data, deviceInstances.Data(), hostVector.Size(), cudaMemcpyHostToDevice);
		}
	}

	~DeviceVector()
	{
		Clear();

		if (m_Data)
			DeviceAllocator<TDevice>::Free(m_Allocator, m_Data);
	}

	void Reset(size_t newCapacity)
	{
		DeviceAllocator<TDevice>::Free(m_Allocator, m_Data);
		m_Data = DeviceAllocator<TDevice>::Alloc(m_Allocator, newCapacity);
		m_Capacity = newCapacity;
	}

	DeviceVector<THost, TDevice>& operator=(const DeviceVector<THost, TDevice>& other)
	{
		if (this != &other)
		{
			Clear();
			m_Allocator = other.m_Allocator;
			m_Capacity = other.m_Capacity;
			Realloc(m_Capacity);
			m_Size = other.Size();
			CudaMemory::CopyAsync<TDevice>(m_Data, other.m_Data, other.m_Size, cudaMemcpyDeviceToDevice);
		}
		return *this;
	}

	DeviceVector<THost, TDevice>& operator=(DeviceVector<THost, TDevice>&& other)
	{
		if (this != &other)
		{
			Clear();
			CudaMemory::Free(m_Data);
			m_Allocator = other.m_Allocator;
			m_Capacity = other.m_Capacity;
			m_Data = other.m_Data;
			m_Size = other.m_Size;
			other.m_Data = nullptr;
		}
		return *this;
	}

	void PushBack(const THost& value)
	{
		if (m_Size >= m_Capacity)
			Realloc(m_Capacity + m_Capacity / 2);

		if constexpr (is_trivially_copyable_to_device<THost>)
			CudaMemory::CopyAsync<TDevice>(m_Data + m_Size, (TDevice*)&value, 1, cudaMemcpyHostToDevice);
		else
		{
			TDevice deviceInstance = THost::ToDevice(value);
			CudaMemory::CopyAsync<TDevice>(m_Data + m_Size, &deviceInstance, 1, cudaMemcpyHostToDevice);
		}
		m_Size++;
	}

	void PopBack()
	{
		assert(m_Size > 0);
		m_Size--;
		if constexpr (!is_trivially_destructible_from_device<THost>)
			THost::DestructFromDevice(DeviceInstance<THost, TDevice>(m_Data + m_Size).Instance());
	}

	void Clear()
	{
		if constexpr (!is_trivially_destructible_from_device<THost>)
		{
			for (size_t i = 0; i < m_Size; i++)
				THost::DestructFromDevice(DeviceInstance<THost, TDevice>(m_Data + i).Instance());
		}

		m_Size = 0;
	}

	size_t Size() const { return m_Size; }

	TDevice* Data() const { return m_Data; }

	DeviceInstance<THost, TDevice> operator[] (size_t index)
	{
		assert(index >= 0 && index < m_Size);
		return DeviceInstance<THost, TDevice>(m_Data + index);
	}

private:
	void Realloc(size_t newCapacity)
	{
		TDevice* newBlock = DeviceAllocator<TDevice>::Alloc(m_Allocator, newCapacity);

		size_t size = std::min(newCapacity, m_Size);

		CudaMemory::CopyAsync<TDevice>(newBlock, m_Data, size, cudaMemcpyDeviceToDevice);

		DeviceAllocator<TDevice>::Free(m_Allocator, m_Data);
		m_Data = newBlock;
		m_Capacity = newCapacity;
	}

private:
	TDevice* m_Data = nullptr;
	DeviceAllocator<TDevice>* m_Allocator = nullptr;

	size_t m_Size = 0;
	size_t m_Capacity = 0;
};

