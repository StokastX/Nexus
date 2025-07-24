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

	__host__ __device__ TangentFrame(float3 n, float3 v0, float3 v1, float3 v2, float2 uv0, float2 uv1, float2 uv2)
		: normal(normalize(n))
	{
		float3 edge0 = v1 - v0;
		float3 edge1 = v2 - v0;
		float2 deltaUV0 = uv1 - uv0;
		float2 deltaUV1 = uv2 - uv0;
		float r = 1.0f / (deltaUV0.x * deltaUV1.y - deltaUV1.x * deltaUV0.y);
		tangent = normalize(r * (deltaUV1.y * edge0 - deltaUV0.y * edge1));
		bitangent = normalize(r * (-deltaUV1.x * edge0 + deltaUV0.x * edge1));
		normal = normalize(cross(tangent, bitangent));
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