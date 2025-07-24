#pragma once
#include <cuda_runtime_api.h>
#include <Utils/cuda_math.h>

struct TangentFrame
{
	TangentFrame() = default;

	// From [Duff et al. 2017] Building an Orthonormal Basis, Revisited
	// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
	__host__ __device__ TangentFrame(float3 n)
		: normal(n)
	{
		float sign = copysignf(1.0f, n.z);
		const float a = -1.0f / (sign + n.z);
		const float b = n.x * n.y * a;
		tangent = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
		bitangent = make_float3(b, sign + n.y * n.y * a, -n.y);
	}

	__host__ __device__ TangentFrame(float3 n, float3 t)
		: normal(n)
	{
		tangent = normalize(t - dot(t, n) * n);
		bitangent = cross(normal, tangent);
	}

	inline __host__ __device__ float3 WorldToLocal(float3 v)
	{
		return make_float3(dot(v, tangent), dot(v, bitangent), dot(v, normal));
	}

	inline __host__ __device__ float3 LocalToWorld(float3 v)
	{
		return tangent * v.x + bitangent * v.y + normal * v.z;
	}

	float3 normal;
	float3 tangent;
	float3 bitangent;
};