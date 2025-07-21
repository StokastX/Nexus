#pragma once

#include <cuda_runtime_api.h>
#include "Ray.cuh"
#include "Geometry/BVH/BVH.h"

static __forceinline__ __device__ float AABBTrace(const D_Ray& ray, float hitDistance, const NXB::AABB& bounds)
{
	float3 bMin = bounds.bMin;
	float3 bMax = bounds.bMax;
	float tx1 = (bMin.x - ray.origin.x) * ray.invDirection.x, tx2 = (bMax.x - ray.origin.x) * ray.invDirection.x;
	float tmin = fmin(tx1, tx2), tmax = fmax(tx1, tx2);
	float ty1 = (bMin.y - ray.origin.y) * ray.invDirection.y, ty2 = (bMax.y - ray.origin.y) * ray.invDirection.y;
	tmin = fmax(tmin, fmin(ty1, ty2)), tmax = fmin(tmax, fmax(ty1, ty2));
	float tz1 = (bMin.z - ray.origin.z) * ray.invDirection.z, tz2 = (bMax.z - ray.origin.z) * ray.invDirection.z;
	tmin = fmax(tmin, fmin(tz1, tz2)), tmax = fmin(tmax, fmax(tz1, tz2));
	if (tmax >= tmin && tmin < hitDistance && tmax > 0) return tmin; else return 1e30f;
}

static __forceinline__ __device__ float AABBTraceWireframe(const D_Ray& ray, float hitDistance, const NXB::AABB& bounds, bool& hitEdge, float3 scale)
{
	float3 bMin = bounds.bMin;
    float3 bMax = bounds.bMax;
    float tx1 = (bMin.x - ray.origin.x) * ray.invDirection.x, tx2 = (bMax.x - ray.origin.x) * ray.invDirection.x;
    float tmin = fmin(tx1, tx2), tmax = fmax(tx1, tx2);
    float ty1 = (bMin.y - ray.origin.y) * ray.invDirection.y, ty2 = (bMax.y - ray.origin.y) * ray.invDirection.y;
    tmin = fmax(tmin, fmin(ty1, ty2)), tmax = fmin(tmax, fmax(ty1, ty2));
    float tz1 = (bMin.z - ray.origin.z) * ray.invDirection.z, tz2 = (bMax.z - ray.origin.z) * ray.invDirection.z;
    tmin = fmax(tmin, fmin(tz1, tz2)), tmax = fmin(tmax, fmax(tz1, tz2));

    if (tmax >= tmin && tmax > 0)
    {
		if (tmin > 0)
		{
			float3 hitPoint = ray.origin + tmin * ray.direction;
			float dist = length(hitPoint - ray.origin);
			int boundaryCount = 0;

			// Check how many coordinates are exactly on the box's boundaries
			if (fabs(hitPoint.x - bMin.x) < 1e-3f * dist || fabs(hitPoint.x - bMax.x) < 1e-3f * dist) boundaryCount++;
			if (fabs(hitPoint.y - bMin.y) < 1e-3f * dist || fabs(hitPoint.y - bMax.y) < 1e-3f * dist) boundaryCount++;
			if (fabs(hitPoint.z - bMin.z) < 1e-3f * dist || fabs(hitPoint.z - bMax.z) < 1e-3f * dist) boundaryCount++;
			if (boundaryCount >= 2)
				hitEdge = true;
		}

		float3 hitPoint = ray.origin + tmax * ray.direction;
		float dist = length(hitPoint - ray.origin);
		int boundaryCount = 0;

		// Check how many coordinates are exactly on the box's boundaries
		if (fabs(hitPoint.x - bMin.x) < 1e-3f * dist || fabs(hitPoint.x - bMax.x) < 1e-3f * dist) boundaryCount++;
		if (fabs(hitPoint.y - bMin.y) < 1e-3f * dist || fabs(hitPoint.y - bMax.y) < 1e-3f * dist) boundaryCount++;
		if (fabs(hitPoint.z - bMin.z) < 1e-3f * dist || fabs(hitPoint.z - bMax.z) < 1e-3f * dist) boundaryCount++;
		if (boundaryCount >= 2)
			hitEdge = true;

		if (tmin < hitDistance)
			return tmin;
    }

    return 1e30f;
}