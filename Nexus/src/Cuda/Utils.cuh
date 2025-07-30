#pragma once

#include <cuda_runtime_api.h>
#include <cudart_platform.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include "Utils/cuda_math.h"

// Sign extend function. See https://github.com/AlanIWBFT/CWBVH/blob/master/src/TraversalKernelCWBVH.cu
inline __device__ uint32_t SignExtendS8x4(uint32_t i) 
{
	uint32_t v; asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(v) : "r"(i)); return v;
}

// Returns max(a, max(b, c))
inline __device__ float vMaxMax(float a, float b, float c)
{
	int result;

	asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

	return __int_as_float(result);
}

// Returns min(a, min(b, c))
inline __device__ float vMinMin(float a, float b, float c)
{
	int result;

	asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));

	return __int_as_float(result);
}

// Extract the i-th byte of x
inline __device__ uint32_t ExtractByte(uint32_t x, uint32_t i)
{
	return (x >> (i * 8)) & 0xff;
}

inline __device__ float Square(const float x)
{
	return x * x;
}

__device__ constexpr float fifthPower(float x)
{
	float r = x * x;
	return r * r * x;
}

__device__ constexpr float sixthPower(const float x)
{
	float r = x * x;
	return r * r * r;
}

template<typename T>
inline __device__ T Barycentric(T t0, T t1, T t2, float2 uv)
{
	return uv.x * t1 + uv.y * t2 + (1.0f - uv.x - uv.y) * t0;
}

constexpr float ORIGIN = 1.0f / 32.0f;
constexpr float FLOAT_SCALE = 1.0f / 65536.0f;
constexpr float INT_SCALE = 256.0f;

// From "A Fast and Robust Method for Avoiding Self-Intersection" (Ray Tracing Gems)
// See https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_6.pdf
// Normal points outward for rays exiting the surface, else is flipped.
inline __device__ float3 OffsetRay(const float3 p, const float3 n)
{
	int3 ofi = { int(INT_SCALE * n.x), int(INT_SCALE * n.y), int(INT_SCALE * n.z) };

	float3 pi = make_float3(
		__int_as_float(__float_as_int(p.x) + ((p.x < 0.0f) ? -ofi.x : ofi.x)),
		__int_as_float(__float_as_int(p.y) + ((p.y < 0.0f) ? -ofi.y : ofi.y)),
		__int_as_float(__float_as_int(p.z) + ((p.z < 0.0f) ? -ofi.z : ofi.z))
	);

	return make_float3(
		fabs(p.x) < ORIGIN ? p.x + FLOAT_SCALE * n.x : pi.x,
		fabs(p.y) < ORIGIN ? p.y + FLOAT_SCALE * n.y : pi.y,
		fabs(p.z) < ORIGIN ? p.z + FLOAT_SCALE * n.z : pi.z
	);
}
