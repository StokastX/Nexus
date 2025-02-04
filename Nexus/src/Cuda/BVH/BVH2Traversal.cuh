#pragma once

#include <cuda_runtime_api.h>
#include "Utils/Utils.h"
#include "Geometry/BVH/BVH.h"
#include "Cuda/Scene/MeshInstance.cuh"
#include "Cuda/Scene/Mesh.cuh"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Cuda/Geometry/Ray.cuh"


__device__ __constant__ NXB::BVH tlas;

inline __device__ void BVH2Trace(D_Mesh* meshes, D_MeshInstance* meshInstances, D_TraceRequestSOA traceRequest, int32_t traceSize, int32_t* traceCount)
{
	D_Mesh mesh;
	NXB::BVH::Node node;
	D_Intersection intersection;

	uint32_t stack[32];
	uint32_t stackPtr = 0;

	uint32_t instanceStackDepth;
	uint32_t instanceIdx;

	D_Ray ray, backupRay;
	int32_t rayIndex;

	bool shouldFetchNewRay = true;

	while (true)
	{
		if (shouldFetchNewRay)
		{
			shouldFetchNewRay = false;

			rayIndex = atomicAdd(traceCount, 1);
			if (rayIndex >= traceSize)
				return;

			mesh.bvh = tlas;
			node = tlas.nodes[mesh.bvh.nodeCount - 1];
			ray = traceRequest.ray.Get(rayIndex);
			backupRay = ray;
			intersection.hitDistance = 1e30f;
			instanceStackDepth = INVALID_IDX;
		}

		if (node.leftChild == INVALID_IDX)
		{
			// Reached a mesh instance
			if (instanceStackDepth == INVALID_IDX)
			{
				instanceIdx = mesh.bvh.primIdx[node.rightChild];
				instanceStackDepth = stackPtr;

				const D_MeshInstance& meshInstance = meshInstances[instanceIdx];
				mesh = meshes[meshInstance.meshIdx];
				node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];

				ray.origin = meshInstance.invTransform.TransformPoint(ray.origin);
				ray.direction = meshInstance.invTransform.TransformVector(ray.direction);
				ray.invDirection = 1.0f / ray.direction;
			}
			// Reached a primitive
			else
			{
				uint32_t triangleIdx = mesh.bvh.primIdx[node.rightChild];
				NXB::Triangle triangle = mesh.triangles[triangleIdx];
				TriangleTrace(triangle, ray, intersection, instanceIdx, triangleIdx);
				if (stackPtr == 0)
				{
					traceRequest.intersection.Set(rayIndex, intersection);
					shouldFetchNewRay = true;
					continue;
				}
				if (stackPtr == instanceStackDepth)
				{
					ray = backupRay;

					mesh.bvh = tlas;
					instanceStackDepth = INVALID_IDX;
				}
				node = mesh.bvh.nodes[stack[--stackPtr]];
			}
			continue;
		}

		NXB::BVH::Node leftChild = mesh.bvh.nodes[node.leftChild];
		NXB::BVH::Node rightChild = mesh.bvh.nodes[node.rightChild];
		float dist1 = AABBTrace(ray, intersection.hitDistance, leftChild.bounds);
		float dist2 = AABBTrace(ray, intersection.hitDistance, rightChild.bounds);

		if (dist1 > dist2)
		{
			Utils::Swap(dist1, dist2);
			Utils::Swap(leftChild, rightChild);
		}

		if (dist1 == 1e30f)
		{
			if (stackPtr == 0)
			{
				traceRequest.intersection.Set(rayIndex, intersection);
				shouldFetchNewRay = true;
				continue;
			}

			if (stackPtr == instanceStackDepth)
			{
				ray = backupRay;

				mesh.bvh = tlas;
				instanceStackDepth = INVALID_IDX;
			}
			node = mesh.bvh.nodes[stack[--stackPtr]];
		}
		else
		{
			if (dist2 != 1e30f)
				stack[stackPtr++] = node.rightChild;

			node = leftChild;
		}
	}
}

inline __device__ void BVH2TraceShadow(D_Mesh* meshes, D_MeshInstance* meshInstances, D_TraceRequestSOA traceRequest, int32_t traceSize, int32_t* traceCount)
{
	D_Mesh mesh;
	NXB::BVH::Node node;
	D_Intersection intersection;

	uint32_t stack[32];
	uint32_t stackPtr = 0;

	uint32_t instanceStackDepth;
	uint32_t instanceIdx;

	D_Ray ray, backupRay;
	int32_t rayIndex;

	bool shouldFetchNewRay = true;

	while (true)
	{
		if (shouldFetchNewRay)
		{
			shouldFetchNewRay = false;

			rayIndex = atomicAdd(traceCount, 1);
			if (rayIndex >= traceSize)
				return;

			mesh.bvh = tlas;
			node = tlas.nodes[mesh.bvh.nodeCount - 1];
			ray = traceRequest.ray.Get(rayIndex);
			backupRay = ray;
			intersection.hitDistance = 1e30f;
			instanceStackDepth = INVALID_IDX;
		}

		if (node.leftChild == INVALID_IDX)
		{
			// Reached a mesh instance
			if (instanceStackDepth == INVALID_IDX)
			{
				instanceIdx = mesh.bvh.primIdx[node.rightChild];
				instanceStackDepth = stackPtr;

				const D_MeshInstance& meshInstance = meshInstances[instanceIdx];
				mesh = meshes[meshInstance.meshIdx];
				node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];

				ray.origin = meshInstance.invTransform.TransformPoint(ray.origin);
				ray.direction = meshInstance.invTransform.TransformVector(ray.direction);
				ray.invDirection = 1.0f / ray.direction;
			}
			// Reached a primitive
			else
			{
				uint32_t triangleIdx = mesh.bvh.primIdx[node.rightChild];
				NXB::Triangle triangle = mesh.triangles[mesh.bvh.primIdx[node.rightChild]];
				TriangleTrace(triangle, ray, intersection, instanceIdx, triangleIdx);
				if (stackPtr == 0)
				{
					traceRequest.intersection.Set(rayIndex, intersection);
					shouldFetchNewRay = true;
					continue;
				}
				if (stackPtr == instanceStackDepth)
				{
					ray = backupRay;

					mesh.bvh = tlas;
					instanceStackDepth = -1;
				}
				node = mesh.bvh.nodes[stack[--stackPtr]];
			}
		}

		NXB::BVH::Node leftChild = mesh.bvh.nodes[node.leftChild];
		NXB::BVH::Node rightChild = mesh.bvh.nodes[node.rightChild];
		float dist1 = AABBTrace(ray, intersection.hitDistance, leftChild.bounds);
		float dist2 = AABBTrace(ray, intersection.hitDistance, rightChild.bounds);

		if (dist1 > dist2)
		{
			Utils::Swap(dist1, dist2);
			Utils::Swap(leftChild, rightChild);
		}

		if (dist1 == 1e30f)
		{
			if (stackPtr == 0)
			{
				traceRequest.intersection.Set(rayIndex, intersection);
				shouldFetchNewRay = true;
				continue;
			}

			if (stackPtr == instanceStackDepth)
			{
				ray = backupRay;

				mesh.bvh = tlas;
				instanceStackDepth = -1;
			}
			node = mesh.bvh.nodes[stack[--stackPtr]];
		}
		else
		{
			if (dist2 != 1e30f)
				stack[stackPtr++] = node.rightChild;

			node = leftChild;
		}
	}
}



//inline __device__ void IntersectBVH2(const D_BVH2& bvh, D_Ray& ray, const uint32_t instanceIdx)
//{
//	D_BVH2Node* node = &bvh.nodes[0], * stack[32];
//	uint32_t stackPtr = 0;
//
//	while (1)
//	{
//		if (node->IsLeaf())
//		{
//			for (uint32_t i = 0; i < node->triCount; i++)
//				bvh.triangles[bvh.triangleIdx[node->leftNode + i]].Trace(ray, instanceIdx, bvh.triangleIdx[node->leftNode + i]);
//
//			if (stackPtr == 0)
//				break;
//			else
//				node = stack[--stackPtr];
//			continue;
//		}
//
//		D_BVH2Node* child1 = &bvh.nodes[node->leftNode];
//		D_BVH2Node* child2 = &bvh.nodes[node->leftNode + 1];
//		float dist1 = D_AABB::IntersectionAABB(ray, child1->aabbMin, child1->aabbMax);
//		float dist2 = D_AABB::IntersectionAABB(ray, child2->aabbMin, child2->aabbMax);
//
//		if (dist1 > dist2)
//		{
//			Utils::Swap(dist1, dist2);
//			Utils::Swap(child1, child2);
//		}
//
//		if (dist1 == 1e30f)
//		{
//			if (stackPtr == 0)
//				break;
//			else
//				node = stack[--stackPtr];
//		}
//		else
//		{
//			node = child1;
//			if (dist2 != 1e30f)
//				stack[stackPtr++] = child2;
//		}
//
//	}
//}