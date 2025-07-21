#pragma once
#include <cuda_runtime_api.h>
#include "Device/CudaMemory.h"


/*
 * Wrapper class holding a device pointer to the device instance.
 * THost is the host class, TDevice is the device class, both can be identical.
 * 
 * If THost implements a static TDevice ToDevice() method, this method 
 * will be used to create a device instance and copy it to the device
 * when using the assignment operator.
 * Likewise, if THost implements a static void DestructFromDevice() method,
 * it will be used for destructing the device instance.
 */
template<typename THost, typename TDevice = THost>
class DeviceInstance
{
public:
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

	DeviceInstance(const DeviceInstance<THost, TDevice>& other)
		: m_Instance(other.m_Instance)
	{
		//assert(is_trivially_copyable_to_device<THost>);
		m_OwnsPtr = true;
		m_DevicePtr = CudaMemory::Allocate<TDevice>(1);
		CudaMemory::CopyAsync<TDevice>(other.m_DevicePtr, m_DevicePtr, 1, cudaMemcpyDeviceToDevice);
	}

	DeviceInstance(DeviceInstance<THost, TDevice>&& other)
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
		DestructDeviceInstance();
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
		if constexpr (!is_trivially_copyable_to_device<THost>)
		{
			TDevice deviceInstance = THost::ToDevice(hostInstance);
			CudaMemory::CopyAsync<TDevice>(m_DevicePtr, &deviceInstance, 1, cudaMemcpyHostToDevice);
			m_Instance = deviceInstance;
		}
		else
		{
			CudaMemory::CopyAsync<TDevice>(m_DevicePtr, (TDevice*)&hostInstance, 1, cudaMemcpyHostToDevice);
			m_Instance = *(TDevice*)&hostInstance;
		}
	}

	void DestructDeviceInstance()
	{
		if constexpr (!is_trivially_destructible_from_device<THost>)
			THost::DestructFromDevice(m_Instance);
	}

private:
	TDevice* m_DevicePtr = nullptr;
	TDevice m_Instance;

	bool m_OwnsPtr = false;
};


