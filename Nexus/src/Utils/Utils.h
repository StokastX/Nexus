#pragma once

#include <cuda_runtime_api.h>
#include <iostream>
#include "cuda_math.h"

#define PI  3.14159265358979323846
#define INV_PI 0.31830988618f
#define TWO_TIMES_PI 6.28318530718f
#define INVALID_IDX (uint32_t)(-1)

#define CheckCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

namespace Utils
{
	template<typename T>
	inline __host__ __device__ void Swap(T& a, T& b) 
	{
		T c = a;
		a = b;
		b = c;
	}

	template<typename T>
	inline __host__ __device__ T SgnE(T val)
	{
		return val < T(0) ? T(-1) : T(1);
	}

	inline __host__ __device__ float ToRadians(float angle)
	{
		return angle * PI / 180.0f;
	}

	inline __host__ __device__ float3 ToRadians(float3 angle)
	{
		return angle * PI / 180.0f;
	}

	inline __host__ __device__ float ToDegrees(float angle)
	{
		return angle * 180.0f / PI;
	}

	inline __host__ __device__ float3 ToDegrees(float3 angle)
	{
		return angle * 180.0f / PI;
	}

	void GetPathAndFileName(const std::string fullPath, std::string& path, std::string& name);
}

/*
 * Checks for an existing implementation of a ToDevice() method using SFINAE.
 * See https://stackoverflow.com/questions/257288/how-can-you-check-whether-a-templated-class-has-a-member-function
 */
template<typename T>
class ImplementsToDevice
{
	typedef char one;
	struct two { char x[2]; };

	template<typename C> static one test(decltype(&C::ToDevice));
	template<typename C> static two test(...);

public:
	enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template<typename T>
class ImplementsDestructFromDevice
{
	typedef char one;
	struct two { char x[2]; };

	template<typename C> static one test(decltype(&C::DestructFromDevice));
	template<typename C> static two test(...);

public:
	enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template<typename T>
constexpr bool is_trivially_copyable_to_device = !ImplementsToDevice<T>::value;

template<typename T>
constexpr bool is_trivially_destructible_from_device = !ImplementsDestructFromDevice<T>::value;


