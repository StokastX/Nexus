#pragma once
#include <type_traits>

/*
 * How a host type is represented on the device.
 *
 * The primary template is the default policy: the type crosses unchanged, as a bitwise copy. It
 * deliberately provides no ToDevice -- under the rule below, a type selecting the primary is
 * exactly a type whose transfer is a memcpy, so a conversion here would be unreachable.
 *
 * A type needing a real conversion opts in from outside, by specialising:
 *
 *     template<>
 *     struct DeviceTraits<Mesh>
 *     {
 *         using DeviceType = D_Mesh;
 *         static D_Mesh ToDevice(const Mesh&);
 *     };
 *
 * Specialising rather than declaring a static THost::ToDevice keeps the mapping expressible for
 * types we do not own (cudaTextureObject_t, NXB::Triangle), and makes the conversion a property of
 * the host/device pair rather than of the host class. The cost is one ODR rule: a specialisation
 * must be visible everywhere DeviceVector<THost> is instantiated, so each one lives immediately
 * below its host type, in the same header.
 */
namespace Nexus {

	template<typename THost>
	struct DeviceTraits
	{
		static_assert(std::is_trivially_copyable_v<THost>,
			"No DeviceTraits specialisation for this type, and it is not trivially copyable: the "
			"bitwise copy the default policy performs would send host pointers to the device. "
			"Specialise DeviceTraits<T> with a DeviceType and a ToDevice(), next to the type.");

		using DeviceType = THost;
	};

	template<typename THost>
	using DeviceType_t = typename DeviceTraits<THost>::DeviceType;

	/*
	 * True when a host array is already a valid device array, so a whole vector can cross in one
	 * cudaMemcpy rather than element by element.
	 *
	 * Derived rather than declared, which imposes an invariant worth stating: a type that needs a
	 * conversion must name a DeviceType distinct from itself. A specialisation that kept
	 * DeviceType = THost and supplied a transforming ToDevice would be classified bulk-copyable
	 * and have its conversion silently skipped. Every conversion here produces a D_ type, so the
	 * invariant holds by construction.
	 */
	template<typename THost>
	inline constexpr bool isBulkCopyable_v =
		std::is_same_v<THost, DeviceType_t<THost>> && std::is_trivially_copyable_v<THost>;

	// Converts one object, whichever policy applies. Only the transfer *strategy* -- one bulk copy
	// against a staging buffer -- still needs to branch on isBulkCopyable_v at the call site.
	template<typename THost>
	DeviceType_t<THost> ConvertToDevice(const THost& host)
	{
		if constexpr (isBulkCopyable_v<THost>)
			return host;
		else
			return DeviceTraits<THost>::ToDevice(host);
	}

}
