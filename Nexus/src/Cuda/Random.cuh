#pragma once
#include <cuda_runtime_api.h>
#include <cudart_platform.h>
#include <device_launch_parameters.h>
#include "Utils/cuda_math.h"

/**
 * Fast random number generator based on Ray Tracing Gems 2 book (cf page 169)
 */

class Random
{
public:
	static inline __device__ unsigned int InitRNG(uint2 pixel, uint2 resolution, unsigned int frameNumber);
	static inline __device__ unsigned int InitRNG(uint32_t index, uint2 resolution, unsigned int frameNumber);
	static inline __device__ float Rand(unsigned int& rngState);
	static inline __device__ float3 RandomCosineHemisphere(unsigned int& rngState);
	static inline __device__ float2 RandomInUnitDisk(unsigned int& rngState);
};

inline __device__ unsigned int jenkinsHash(unsigned int x)
{
	x += x << 10;
	x ^= x >> 6;
	x += x << 3;
	x ^= x >> 11;
	x += x << 15;
	return x;
}

// PCG version
inline __device__ uint4 pcg4d(uint4 v)
{
	v = v * 1664525u + 1013904223u;

	v.x += v.y * v.w;
	v.y += v.z * v.x; 
	v.z += v.x * v.y; 
	v.w += v.y * v.z;

	v.x ^= v.x >> 16u;
	v.y ^= v.y >> 16u;
	v.z ^= v.z >> 16u;
	v.w ^= v.w >> 16u;

	v.x += v.y * v.w; 
	v.y += v.z * v.x; 
	v.z += v.x * v.y; 
	v.w += v.y * v.z;

	return v;
}

inline __device__ unsigned int xorShift(unsigned int& rngState)
{
	rngState ^= rngState << 13;
	rngState ^= rngState >> 17;
	rngState ^= rngState << 5;
	return rngState;
}

inline __device__ float uintToFloat(unsigned int x)
{
	return __uint_as_float(0x3f800000 | (x >> 9)) - 1.0f;
}

inline __device__ unsigned int Random::InitRNG(uint2 pixel, uint2 resolution, unsigned int frameNumber)
{
	unsigned int rngState = dot(pixel, make_uint2(1, resolution.x)) ^ jenkinsHash(frameNumber);
	if (rngState == 0)
		rngState = 1;
	return jenkinsHash(rngState);
}

inline __device__ unsigned int Random::InitRNG(uint32_t index, uint2 resolution, unsigned int frameNumber)
{
	return Random::InitRNG(make_uint2(1, index), resolution, frameNumber);
}

inline __device__ float Random::Rand(unsigned int& rngState)
{
	return uintToFloat(xorShift(rngState));
}

inline __device__ float3 Random::RandomCosineHemisphere(unsigned int& rngState)
{
	float r1 = Rand(rngState);
	float r2 = Rand(rngState);
	float B = sqrt(r2);

	float phi = 2 * PI * r1;
	float x = cos(phi) * B;
	float y = sin(phi) * B;
	float z = sqrt(1 - r2);

	return make_float3(x, y, z);
}

// From https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
// Warning: this distorts areas on the disk and should not be used for stratified sampling
inline __device__ float2 Random::RandomInUnitDisk(unsigned int& rngState)
{
	float2 u = make_float2(Rand(rngState), Rand(rngState));
	float r = sqrt(u.x);
	float theta = 2 * PI * u.y;
	return make_float2(r * cos(theta), r * sin(theta));
}
