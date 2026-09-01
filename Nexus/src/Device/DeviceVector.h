#pragma once
#include <vector>
#include <cassert>
#include "CudaMemory.h"
#include "DeviceTraits.h"

/*
 * Device array of the device representation of THost, as named by DeviceTraits<THost>.
 *
 * No constructor or destructor ever runs on an element -- an object cannot be constructed on the
 * GPU from the host -- so an element is either bitwise-copied or passed through the
 * specialisation ToDevice on its way over. Elements never own device memory: everything that
 * allocates is an RAII type on the host side (DeviceVector itself, CudaTexture), which is why
 * clearing this vector is only a size reset.
 */
namespace Nexus {

	template<typename THost>
	class DeviceVector
	{
	public:
		using TDevice = DeviceType_t<THost>;

		DeviceVector()
		{
			Realloc(2);
		}

		DeviceVector(size_t size)
		{
			Realloc(size);
			m_Size = size;
		}

		// Move-only: this owns a device allocation, and nothing ever copied one.
		DeviceVector(const DeviceVector<THost>& other) = delete;
		DeviceVector<THost>& operator=(const DeviceVector<THost>& other) = delete;

		DeviceVector(DeviceVector<THost>&& other)
			: m_Capacity(other.m_Capacity), m_Size(other.m_Size), m_Data(other.m_Data)
		{
			other.m_Data = nullptr;
		}

		DeviceVector(const std::vector<THost>& hostVector)
		{
			Realloc(hostVector.size());
			m_Size = hostVector.size();

			// The one place the policy changes the transfer strategy rather than only the value: a
			// bulk-copyable host array is already a valid device array, so it crosses in one copy.
			if constexpr (isBulkCopyable_v<THost>)
				CudaMemory::CopyAsync<TDevice>(m_Data, (TDevice*)hostVector.data(), hostVector.size(), cudaMemcpyHostToDevice);
			else
			{
				std::vector<TDevice> deviceInstances(hostVector.size());
				for (size_t i = 0; i < hostVector.size(); i++)
					deviceInstances[i] = DeviceTraits<THost>::ToDevice(hostVector[i]);

				CudaMemory::CopyAsync<TDevice>(m_Data, deviceInstances.data(), hostVector.size(), cudaMemcpyHostToDevice);
			}
		}

		~DeviceVector()
		{
			Clear();

			if (m_Data)
				CudaMemory::Free(m_Data);
		}

		DeviceVector<THost>& operator=(DeviceVector<THost>&& other)
		{
			if (this != &other)
			{
				Clear();

				// Released before the pointer is overwritten, or the old block leaks.
				CudaMemory::Free(m_Data);

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

			TDevice deviceInstance = ConvertToDevice(value);
			CudaMemory::CopyAsync<TDevice>(m_Data + m_Size, &deviceInstance, 1, cudaMemcpyHostToDevice);
			m_Size++;
		}

		void PushBack(THost&& value)
		{
			if (m_Size >= m_Capacity)
				Realloc(m_Capacity + m_Capacity / 2);

			TDevice deviceInstance = ConvertToDevice(value);
			CudaMemory::CopyAsync<TDevice>(m_Data + m_Size, &deviceInstance, 1, cudaMemcpyHostToDevice);
			m_Size++;
		}

		void PopBack()
		{
			assert(m_Size > 0);
			m_Size--;
		}

		void Clear()
		{
			m_Size = 0;
		}

		size_t Size() const { return m_Size; }

		TDevice* Data() const { return m_Data; }

	private:
		void Realloc(size_t newCapacity)
		{
			TDevice* newBlock = CudaMemory::Allocate<TDevice>(newCapacity);

			size_t size = std::min(newCapacity, m_Size);

			// m_Data is null on the first Realloc, from the default constructor.
			if (m_Data && size > 0)
				CudaMemory::CopyAsync<TDevice>(newBlock, m_Data, size, cudaMemcpyDeviceToDevice);

			CudaMemory::Free(m_Data);
			m_Data = newBlock;
			m_Capacity = newCapacity;
		}

	private:
		TDevice* m_Data = nullptr;

		size_t m_Size = 0;
		size_t m_Capacity = 0;
	};

}
