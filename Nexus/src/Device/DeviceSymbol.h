#pragma once
#include <cuda_runtime_api.h>
#include "Device/CudaMemory.h"
#include "Device/DeviceTraits.h"

/*
 * A typed, non-owning handle to one device global.
 *
 * Every instance in the codebase is built from the address of a __device__ variable the kernels
 * read -- the scene, the TLAS, the queue sizes, the accumulation buffer pointer. Assigning a host
 * value writes through to that variable, converting on the way via DeviceTraits<THost>.
 *
 * The class also keeps a host copy of what it last wrote. That is not decoration: PathTracer's
 * teardown recovers the device pointers stored inside the SOA symbols from it, and the pixel query
 * reads a result the kernel wrote via Synchronize().
 *
 * It owns nothing. This was previously DeviceInstance, which could also allocate a device object
 * of its own -- but nothing ever constructed one that way, so the allocating and copying paths
 * were dead and the name described a role the class never played.
 */
namespace Nexus {

	template<typename THost>
	class DeviceSymbol
	{
	public:
		using TDevice = DeviceType_t<THost>;

		DeviceSymbol() = default;

		DeviceSymbol(TDevice* symbolAddress)
			: m_SymbolAddress(symbolAddress)
		{
		}

		// Writes the host value through to the device global, and records what was written.
		void operator=(const THost& hostInstance)
		{
			TDevice deviceValue = ConvertToDevice(hostInstance);
			CudaMemory::CopyAsync<TDevice>(m_SymbolAddress, &deviceValue, 1, cudaMemcpyHostToDevice);
			m_Value = deviceValue;
		}

		/*
		 * The last value written, or whatever Synchronize() last read back -- never a fresh read.
		 *
		 * const on purpose. A mutable pointer here would let `symbol->field = x` compile, which
		 * would modify the host copy alone and leave the device global untouched, with nothing to
		 * say so. Assign to the symbol instead.
		 */
		const TDevice* operator->() const { return &m_Value; }
		const TDevice& Value() const { return m_Value; }

		// The address of the global itself, for callers that write it by other means.
		TDevice* Data() const { return m_SymbolAddress; }

		// Reads the global back, for symbols a kernel writes rather than reads.
		void Synchronize()
		{
			CudaMemory::Copy(&m_Value, m_SymbolAddress, 1, cudaMemcpyDeviceToHost);
		}

	private:
		TDevice* m_SymbolAddress = nullptr;

		// Value-initialised rather than read back at construction.
		TDevice m_Value{};
	};

}
