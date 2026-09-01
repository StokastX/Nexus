#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include "Device/CudaMemory.h"
#include "Device/DeviceTraits.h"
#include "Device/DeviceVector.h"

/*
 * A host array, its device mirror, and the set of elements that differ between them.
 *
 * The pattern this replaces was written out by hand five times -- materials, mesh instances,
 * lights, meshes, textures -- as a std::vector, a DeviceVector and a std::set of dirty indices,
 * each with its own InvalidateX() that every editor call site had to remember to call.
 *
 * The vector hands out const references for reading and a MutationGuard for writing, so there is
 * no way to modify an element without the change being noticed.
 *
 * Elements are never destroyed on the device: under the ownership rule in DeviceTraits.h, device
 * memory belongs to an RAII owner on the host side, and a device element is only a flat view of
 * it. Removing an element therefore drops the view, never the memory.
 */
namespace Nexus {

	template<typename THost>
	class MirroredVector;


	/*
	 * Write handle to one element, which marks it dirty if the write actually changed anything.
	 *
	 * The comparison is made on the *device* representation rather than the host object. That is
	 * the question the dirty set exists to answer -- would the GPU see a different value? -- and it
	 * is the only form of the question that works for every type: a D_ struct is always trivially
	 * copyable, whereas MeshInstance holds a std::string and Mesh holds three device containers.
	 * Renaming a MeshInstance correctly does not dirty it, because name is not in D_MeshInstance.
	 *
	 * Non-copyable and non-movable on purpose: a duplicated guard would run the comparison twice,
	 * and one outliving its statement would compare against a stale snapshot. C++17 guaranteed
	 * copy elision is what lets Mutate() still return one by value.
	 */
	template<typename THost>
	class MutationGuard
	{
	public:
		using TDevice = DeviceType_t<THost>;

		MutationGuard(MirroredVector<THost>& owner, size_t index);
		~MutationGuard();

		MutationGuard(const MutationGuard&) = delete;
		MutationGuard& operator=(const MutationGuard&) = delete;
		MutationGuard(MutationGuard&&) = delete;
		MutationGuard& operator=(MutationGuard&&) = delete;

		THost* operator->() const;
		THost& operator*() const;

	private:
		MirroredVector<THost>* m_Owner;
		size_t m_Index;

		// The device form of the element as it stood when the guard was taken.
		TDevice m_Before;
	};


	template<typename THost>
	class MirroredVector
	{
	public:
		using TDevice = DeviceType_t<THost>;

		// memcmp on the device form is what decides whether to re-upload. A D_ type is bound for
		// the GPU, so it is always trivially copyable; this only states the requirement out loud.
		static_assert(std::is_trivially_copyable_v<TDevice>,
			"The device representation must be trivially copyable to be compared bytewise");

		MirroredVector() = default;

		// ---- reading ----------------------------------------------------------------------

		const std::vector<THost>& Host() const { return m_Host; }

		const THost& operator[](size_t index) const
		{
			assert(index < m_Host.size());
			return m_Host[index];
		}

		size_t Size() const { return m_Host.size(); }
		bool Empty() const { return m_Host.empty(); }

		// The device array the kernels index. Its address changes when the vector grows, so
		// anything caching Data() -- a __constant__ symbol, a field of D_Scene -- must be
		// refreshed after any PushBack.
		DeviceVector<THost>& Device() { return m_Device; }
		const DeviceVector<THost>& Device() const { return m_Device; }
		TDevice* DeviceData() const { return m_Device.Data(); }

		// ---- writing ----------------------------------------------------------------------

		/*
		 * The only way to obtain a writable element:
		 *
		 *     auto light = scene.Lights().Mutate(idx);
		 *     ImGui::DragFloat("Intensity", &light->point.intensity);
		 *
		 * The guard must be a named local. Binding it to a temporary that dies at the end of the
		 * full expression would snapshot and compare before the edit happens.
		 */
		MutationGuard<THost> Mutate(size_t index);

		/*
		 * Marks an element dirty without going through a guard.
		 *
		 * The escape hatch for bulk edits, where taking one guard per element would convert every
		 * element twice for no benefit -- a loader writing transforms over a whole array, say.
		 * Prefer Mutate() anywhere someone could forget this call.
		 */
		void MarkDirty(size_t index)
		{
			assert(index < m_Host.size());
			m_Dirty.insert(static_cast<uint32_t>(index));
		}

		/*
		 * The elements changed since the last Flush.
		 *
		 * Read-only, and deliberately exposed: some state is *derived* from these elements and has
		 * to be recomputed when they change -- mesh lights from material emission, for one. The
		 * encapsulation that matters is that nothing can be modified without being recorded here;
		 * being able to see what was recorded does not weaken it.
		 *
		 * Read it before the Flush that clears it.
		 */
		const std::set<uint32_t>& DirtyIndices() const { return m_Dirty; }

		void MarkAllDirty()
		{
			for (size_t i = 0; i < m_Host.size(); i++)
				MarkDirty(i);
		}

		// ---- growing and shrinking ---------------------------------------------------------

		size_t PushBack(const THost& value)
		{
			m_Host.push_back(value);

			// Converted from its final resting place, not from the argument: for a type whose
			// device form points into memory the element owns, those pointers must be read after
			// the element has been stored.
			m_Device.PushBack(m_Host.back());
			m_StructureChanged = true;
			return m_Host.size() - 1;
		}

		size_t PushBack(THost&& value)
		{
			m_Host.push_back(std::move(value));
			m_Device.PushBack(m_Host.back());
			m_StructureChanged = true;
			return m_Host.size() - 1;
		}

		template<typename... TArgs>
		size_t EmplaceBack(TArgs&&... args)
		{
			m_Host.emplace_back(std::forward<TArgs>(args)...);
			m_Device.PushBack(m_Host.back());
			m_StructureChanged = true;
			return m_Host.size() - 1;
		}

		/*
		 * Removes one element from both sides.
		 *
		 * This is the whole reason the pair lives in one type. Erasing from the host vector alone
		 * leaves the device array a different length, with every element past the removal at the
		 * wrong index -- and any dirty index past it naming a different object than it did when it
		 * was recorded. Getting that right once, here, is what stops it being got wrong at each
		 * call site.
		 */
		void Erase(size_t index)
		{
			assert(index < m_Host.size());

			m_Host.erase(m_Host.begin() + index);
			m_Device.PopBack();

			// Everything from the removal onwards has shifted down one slot, so its device copy is
			// stale. Re-uploading the tail is simpler than renumbering, and these arrays are small.
			for (size_t i = index; i < m_Host.size(); i++)
				Upload(i);

			// The tail is now clean, and the recorded indices into it no longer mean what they did.
			m_Dirty.erase(m_Dirty.lower_bound(static_cast<uint32_t>(index)), m_Dirty.end());

			m_StructureChanged = true;
		}

		void Clear()
		{
			m_Host.clear();
			m_Device.Clear();
			m_Dirty.clear();
			m_StructureChanged = true;
		}

		// ---- synchronising -----------------------------------------------------------------

		/*
		 * Whether anything about the array has changed since the last Flush.
		 *
		 * Adding or removing an element counts, even though PushBack and Erase already wrote the
		 * device side themselves. Consumers care about more than the element bytes: the TLAS is
		 * built from the whole instance array, so it has to be rebuilt when one appears or
		 * disappears, not only when one is edited. Without this, loading a model would leave the
		 * array dirty-free and the TLAS never built.
		 */
		bool Dirty() const { return !m_Dirty.empty() || m_StructureChanged; }

		void Flush()
		{
			for (uint32_t i : m_Dirty)
				Upload(i);

			m_Dirty.clear();
			m_StructureChanged = false;
		}

	private:
		/*
		 * Writes one element's device form straight into the device array.
		 *
		 * deviceValue is a local, which is safe because the source is pageable host memory:
		 * cudaMemcpyAsync stages such a copy before returning, so it does not outlive this frame.
		 */
		void Upload(size_t index)
		{
			TDevice deviceValue = ConvertToDevice(m_Host[index]);
			CudaMemory::CopyAsync<TDevice>(m_Device.Data() + index, &deviceValue, 1, cudaMemcpyHostToDevice);
		}

		friend class MutationGuard<THost>;

		THost& Raw(size_t index) { return m_Host[index]; }

	private:
		std::vector<THost> m_Host;
		DeviceVector<THost> m_Device;

		// Ordered, which is what lets Erase drop a whole tail of indices in one call.
		std::set<uint32_t> m_Dirty;

		// Set by anything that changes the array's length. See Dirty().
		bool m_StructureChanged = false;
	};


	// ---- MutationGuard, defined once MirroredVector is complete ----------------------------

	template<typename THost>
	MutationGuard<THost>::MutationGuard(MirroredVector<THost>& owner, size_t index)
		: m_Owner(&owner), m_Index(index), m_Before(ConvertToDevice(owner.Host()[index]))
	{
	}

	template<typename THost>
	MutationGuard<THost>::~MutationGuard()
	{
		const TDevice after = ConvertToDevice(m_Owner->Host()[m_Index]);

		if (std::memcmp(&m_Before, &after, sizeof(TDevice)) != 0)
			m_Owner->MarkDirty(m_Index);
	}

	// Resolved through the index rather than a cached pointer, so a PushBack that reallocates the
	// host vector during the guard's lifetime cannot leave it pointing at freed storage.
	template<typename THost>
	THost* MutationGuard<THost>::operator->() const { return &m_Owner->Raw(m_Index); }

	template<typename THost>
	THost& MutationGuard<THost>::operator*() const { return m_Owner->Raw(m_Index); }


	template<typename THost>
	MutationGuard<THost> MirroredVector<THost>::Mutate(size_t index)
	{
		assert(index < m_Host.size());
		return MutationGuard<THost>(*this, index);
	}

}
