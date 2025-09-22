#pragma once
#include "Memory/Host/Vector.h"
#include "Memory/Device/DeviceVector.h"
#include "Memory/SynchronizedInstance.h"

/*
 * Vector class that contains a host side Vector, and a device side Vector
 */
namespace Nexus {

	template<typename THost, typename TDevice = THost>
	class SynchronizedVector
	{
	public:
		SynchronizedVector() = default;

		SynchronizedVector(const Vector<THost>& hostVector, DeviceAllocator* deviceAllocator = nullptr)
			: m_HostVector(hostVector), m_DeviceVector(hostVector, deviceAllocator) {
		}

		SynchronizedVector(const SynchronizedVector<THost, TDevice>& other)
			: m_HostVector(other.m_HostVector, hostAllocator), m_DeviceVector(other.m_DeviceVector, deviceAllocator) {
		}

		void PushBack(const THost& value)
		{
			m_HostVector.PushBack(value);
			m_DeviceVector.PushBack(value);
		}

		void PushBack(THost&& value)
		{
			m_HostVector.PushBack(value);
			m_DeviceVector.PushBack(value);
		}

		void PopBack()
		{
			m_HostVector.PopBack();
			m_DeviceVector.PopBack();
		}

		void Clear()
		{
			m_HostVector.Clear();
			m_DeviceVector.Clear();
		}

		size_t Size()
		{
			assert(m_HostVector.Size() == m_DeviceVector.Size());
			return m_HostVector.Size();
		}

		SynchronizedInstance<THost, TDevice> operator[] (size_t index)
		{
			assert(index > 0 && index < m_Size);
			return SynchronizedInstance<THost, TDevice>(m_HostVector[index], m_DeviceVector.Data() + index);
		}

		THost* HostData() { return m_HostVector.Data(); }
		TDevice* DeviceData() { return m_Device.Data(); }

	private:
		Vector<THost> m_HostVector;
		DeviceVector<THost, TDevice> m_DeviceVector;
	};

}