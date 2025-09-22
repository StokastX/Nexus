#pragma once

#include "Device/DeviceInstance.h"

namespace Nexus {

	template<typename THost, typename TDevice = THost>
	class SynchronizedInstance
	{
		SynchronizedInstance() = delete;

		SynchronizedInstance(THost& hostInstance, TDevice* deviceInstance)
			: m_DeviceInstance(deviceInstance), m_HostInstance(hostInstance) {
		}

		void operator=(const THost& hostInstance)
		{
			m_HostInstance = hostInstance;
			m_DeviceInstance = hostInstance;
		}

	private:
		DeviceInstance<THost, TDevice> m_DeviceInstance;
		THost& m_HostInstance;
	};

}