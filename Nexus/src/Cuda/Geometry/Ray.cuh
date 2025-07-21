#pragma once

#include "Cuda/Scene/Material.cuh"

struct D_Intersection
{
	// Ray hit distance
	float hitDistance = 1.0e30f;
	// Barycentric coordinates;
	float u, v;

	uint32_t triIdx;
	uint32_t instanceIdx;
};

struct D_IntersectionSOA
{
	// Ray hit distance
	float* hitDistance;
	// Barycentric coordinates;
	float* u;
	float* v;

	uint32_t* triIdx;
	uint32_t* instanceIdx;

	inline __device__ D_Intersection Get(uint32_t index)
	{
		D_Intersection intersection = {
			hitDistance[index],
			u[index],
			v[index],
			triIdx[index],
			instanceIdx[index]
		};
		return intersection;
	}
	inline __device__ void Set(uint32_t index, D_Intersection intersection)
	{
		hitDistance[index] = intersection.hitDistance;
		u[index] = intersection.u;
		v[index] = intersection.v;
		triIdx[index] = intersection.triIdx;
		instanceIdx[index] = intersection.instanceIdx;
	}
};

struct D_Ray
{
	__device__ D_Ray() = default;
	__device__ D_Ray(float3 o, float3 d)
		:origin(o), direction(d), invDirection(1.0f / direction) {};

	// Ray origin
	float3 origin = make_float3(0.0f);

	// Ray direction
	float3 direction = make_float3(0.0f, 0.0f, 1.0f);

	// Ray inverse direction (reduce divisions for optimization)
	float3 invDirection = make_float3(0.0f, 0.0f, 1.0f);

	//D_Intersection hit;

	inline  __device__ float3 PointAtParameter(float t) const { return origin + direction * t; };
};

struct D_RaySOA
{
	float3* origin;
	float3* direction;

	inline __device__ D_Ray Get(uint32_t index)
	{
		return D_Ray(origin[index], direction[index]);
	}

	inline __device__ void Set(uint32_t index, const D_Ray& ray)
	{
		origin[index] = ray.origin;
		direction[index] = ray.direction;
	}
};

