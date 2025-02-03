#pragma once

#include <cuda_runtime_api.h>
#include "Utils/Utils.h"
#include "BVH2.cuh"
#include "BVHInstance.cuh"

__device__ __constant__ D_BVH2 tlas2;
__device__ __constant__ D_BVHInstance* blas2;
__device__ __constant__ D_BVH2* bvhs2;

inline __device__ void IntersectBVH2(const D_BVH2& bvh, D_Ray& ray, const uint32_t instanceIdx)
{
	D_BVH2Node* node = &bvh.nodes[0], * stack[32];
	uint32_t stackPtr = 0;

	while (1)
	{
		if (node->IsLeaf())
		{
			for (uint32_t i = 0; i < node->triCount; i++)
				bvh.triangles[bvh.triangleIdx[node->leftNode + i]].Trace(ray, instanceIdx, bvh.triangleIdx[node->leftNode + i]);

			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];
			continue;
		}

		D_BVH2Node* child1 = &bvh.nodes[node->leftNode];
		D_BVH2Node* child2 = &bvh.nodes[node->leftNode + 1];
		float dist1 = D_AABB::IntersectionAABB(ray, child1->aabbMin, child1->aabbMax);
		float dist2 = D_AABB::IntersectionAABB(ray, child2->aabbMin, child2->aabbMax);

		if (dist1 > dist2)
		{
			Utils::Swap(dist1, dist2);
			Utils::Swap(child1, child2);
		}

		if (dist1 == 1e30f)
		{
			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f)
				stack[stackPtr++] = child2;
		}

	}
}