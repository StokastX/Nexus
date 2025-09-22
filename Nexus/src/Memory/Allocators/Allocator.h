#pragma once
#include <iostream>

namespace Nexus {

	template<typename T>
	class Allocator
	{
	public:
		Allocator() = default;

		static T* Alloc(Allocator* allocator, size_t count)
		{
			if (allocator)
				return allocator->Alloc(count);
			else
				return (T*)::operator new(count * sizeof(T));
		}

		static void Free(Allocator* allocator, T* ptr)
		{
			if (allocator)
				allocator->Free(ptr);
			else
				::operator delete(ptr);
		}

	protected:
		virtual T* Alloc(size_t count)
		{
			return (T*)::operator new(count * sizeof(T));
		}
		virtual void Free(T* ptr)
		{
			::operator delete(ptr);
		}
	};

}