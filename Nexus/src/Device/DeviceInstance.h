#pragma once
#include <cuda_runtime_api.h>
#include "Device/CudaMemory.h"
#include "Device/DeviceTraits.h"


/*
 * Wrapper class holding a device pointer to one device instance.
 *
 * The device representation comes from DeviceTraits<THost>: THost itself when it crosses as a
 * bitwise copy, otherwise the DeviceType that its specialisation names.
 */
namespace Nexus {

	template<typename THost>
	class DeviceInstance
	{
	public:
		using TDevice = DeviceType_t<THost>;

		DeviceInstance() = default;

		DeviceInstance(TDevice* devicePtr)
			: m_DevicePtr(devicePtr), m_OwnsPtr(false)
		{
			m_Instance = Get();
		}

		DeviceInstance(const THost& hostInstance)
			: m_OwnsPtr(true)
		{
			m_DevicePtr = CudaMemory::Allocate<TDevice>(1);
			SetDeviceInstance(hostInstance);
		}

		DeviceInstance(const DeviceInstance<THost>& other)
			: m_Instance(other.m_Instance)
		{
			m_OwnsPtr = true;
			m_DevicePtr = CudaMemory::Allocate<TDevice>(1);
			CudaMemory::CopyAsync<TDevice>(other.m_DevicePtr, m_DevicePtr, 1, cudaMemcpyDeviceToDevice);
		}

		DeviceInstance(DeviceInstance<THost>&& other)
			: m_OwnsPtr(other.m_OwnsPtr), m_DevicePtr(other.m_DevicePtr), m_Instance(other.m_Instance)
		{
			other.m_DevicePtr = nullptr;
		}

		~DeviceInstance()
		{
			if (m_OwnsPtr && m_DevicePtr)
			{
				CudaMemory::Free(m_DevicePtr);
				m_DevicePtr = nullptr;
			}
		}

		void operator=(const THost& hostInstance)
		{
			SetDeviceInstance(hostInstance);
		}


		TDevice* operator->()
		{
			// Get the instance from copyAsync
			return &m_Instance;
		}

		TDevice Instance() { return m_Instance; }

		TDevice* Data() { return m_DevicePtr; }

		// Get the latest instance from device
		void Synchronize() { m_Instance = Get(); }

	private:

		TDevice Get()
		{
			TDevice target;
			CudaMemory::Copy(&target, m_DevicePtr, 1, cudaMemcpyDeviceToHost);
			return target;
		}

		void SetDeviceInstance(const THost& hostInstance)
		{
			TDevice deviceInstance = ConvertToDevice(hostInstance);
			CudaMemory::CopyAsync<TDevice>(m_DevicePtr, &deviceInstance, 1, cudaMemcpyHostToDevice);
			m_Instance = deviceInstance;
		}

	private:
		TDevice* m_DevicePtr = nullptr;
		TDevice m_Instance;

		bool m_OwnsPtr = false;
	};

}
