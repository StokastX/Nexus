#pragma once

#include <iostream>
#include "Memory/Allocators/Allocator.h"
#include <string.h>

/*
 * Vector class based on The Cherno's video. See https://www.youtube.com/watch?v=ryRf4Jh_YC0
 */

template<typename T>
class Vector
{
public:
	Vector()
	{
		Realloc(2);
	}

	Vector(size_t size, Allocator<T>* allocator = nullptr)
		:m_Allocator(allocator)
	{
		Realloc(size);
		m_Size = size;
	}

	Vector(const Vector<T>& other)
		:m_Allocator(other.m_Allocator)
	{
		Realloc(m_Size);
		m_Size = other.Size();

		if (std::is_trivially_copyable_v<T>)
			memcpy(m_Data, other.Data(), m_Size * sizeof(T));
		else
		{
			for (size_t i = 0; i < m_Size; i++)
				m_Data[i] = other[i];
		}
	}

	Vector(Vector<T>&& other)
		: m_Allocator(other.m_Allocator), m_Capacity(other.m_Capacity), m_Size(other.m_Size), m_Data(other.m_Data)
	{
		other.m_Data = nullptr;
	}

	~Vector()
	{
		Clear();
		Allocator<T>::Free(m_Allocator, m_Data);
	}

	void PushBack(const T& value)
	{
		if (m_Size >= m_Capacity)
			Realloc(m_Capacity + m_Capacity / 2);

		m_Data[m_Size++] = value;
	}

	void PushBack(T&& value)
	{
		if (m_Size >= m_Capacity)
			Realloc(m_Capacity + m_Capacity / 2);

		m_Data[m_Size++] = std::move(value);
	}

	template<typename... Args>
	T& EmplaceBack(Args&&... args)
	{
		if (m_Size >= m_Capacity)
			Realloc(m_Capacity + m_Capacity / 2);

		new(&m_Data[m_Size]) T(std::forward<Args>(args)...);
		return m_Data[m_Size++];
	}

	void PopBack()
	{
		assert(m_Size > 0);
		m_Data[--m_Size].~T();
	}

	void Clear()
	{
		if (!std::is_trivially_destructible_v<T>)
		{
			for (size_t i = 0; i < m_Size; i++)
				m_Data[i].~T();
		}

		m_Size = 0;
	}

	size_t Size() const { return m_Size; }

	T* Data() const { return m_Data; }

	const T& operator[] (size_t index) const 
	{
		assert(index >= 0 && index < m_Size);
		return m_Data[index]; 
	}

	T& operator[] (size_t index)
	{
		assert(index >= 0 && index < m_Size);
		return m_Data[index]; 
	}

private:
	void Realloc(size_t newCapacity)
	{
		T* newBlock = Allocator<T>::Alloc(m_Allocator, newCapacity);

		size_t size = std::min(newCapacity, m_Size);

		if (std::is_trivially_copyable_v<T>)
			memcpy(newBlock, m_Data, size * sizeof(T));

		else
		{
			for (size_t i = 0; i < size; i++)
				new(&newBlock[i]) T(std::move(m_Data[i]));
		}

		for (size_t i = 0; i < size; i++)
			m_Data[i].~T();

		Allocator<T>::Free(m_Allocator, m_Data);
		m_Data = newBlock;
		m_Capacity = newCapacity;
	}

private:
	T* m_Data = nullptr;
	Allocator<T>* m_Allocator = nullptr;

	size_t m_Size = 0;
	size_t m_Capacity = 0;
};