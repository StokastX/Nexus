#pragma once

#include <cuda_runtime_api.h>
#include "Utils/Utils.h"
#include "Geometry/BVH/BVH.h"
#include "Cuda/Scene/MeshInstance.cuh"
#include "Cuda/Scene/Mesh.cuh"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Cuda/Geometry/Ray.cuh"


//__device__ __constant__ NXB::BVH tlas;
//
//inline __device__ void BVH2Trace(D_Mesh* meshes, D_MeshInstance* meshInstances, D_TraceRequestSOA traceRequest, int32_t traceSize, int32_t* traceCount)
//{
//	D_Mesh mesh;
//	NXB::BVH::Node node;
//	D_Intersection intersection;
//
//	uint32_t stack[32];
//	uint32_t stackPtr = 0;
//
//	uint32_t instanceStackDepth;
//	uint32_t instanceIdx;
//
//	D_Ray ray, backupRay;
//	int32_t rayIndex;
//
//	bool shouldFetchNewRay = true;
//
//	while (true)
//	{
//		if (shouldFetchNewRay)
//		{
//			shouldFetchNewRay = false;
//
//			rayIndex = atomicAdd(traceCount, 1);
//			if (rayIndex >= traceSize)
//				return;
//
//			mesh.bvh = tlas;
//			node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];
//			ray = traceRequest.ray.Get(rayIndex);
//			backupRay = ray;
//			intersection.hitDistance = 1e30f;
//			instanceStackDepth = INVALID_IDX;
//		}
//
//		if (node.leftChild == INVALID_IDX)
//		{
//			// Reached a mesh instance
//			if (instanceStackDepth == INVALID_IDX)
//			{
//				instanceIdx = node.rightChild;
//				instanceStackDepth = stackPtr;
//
//				const D_MeshInstance& meshInstance = meshInstances[instanceIdx];
//				mesh = meshes[meshInstance.meshIdx];
//				node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];
//
//				ray.origin = meshInstance.invTransform.TransformPoint(ray.origin);
//				ray.direction = meshInstance.invTransform.TransformVector(ray.direction);
//				ray.invDirection = 1.0f / ray.direction;
//			}
//			// Reached a primitive
//			else
//			{
//				uint32_t triangleIdx = node.rightChild;
//				NXB::Triangle triangle = mesh.triangles[triangleIdx];
//				TriangleTrace(triangle, ray, intersection, instanceIdx, triangleIdx);
//				if (stackPtr == 0)
//				{
//					traceRequest.intersection.Set(rayIndex, intersection);
//					shouldFetchNewRay = true;
//					continue;
//				}
//				if (stackPtr == instanceStackDepth)
//				{
//					ray = backupRay;
//
//					mesh.bvh = tlas;
//					instanceStackDepth = INVALID_IDX;
//				}
//				node = mesh.bvh.nodes[stack[--stackPtr]];
//			}
//			continue;
//		}
//
//		NXB::BVH::Node leftChild = mesh.bvh.nodes[node.leftChild];
//		NXB::BVH::Node rightChild = mesh.bvh.nodes[node.rightChild];
//		float dist1 = AABBTrace(ray, intersection.hitDistance, leftChild.bounds);
//		float dist2 = AABBTrace(ray, intersection.hitDistance, rightChild.bounds);
//
//		if (dist1 > dist2)
//		{
//			Utils::Swap(dist1, dist2);
//			Utils::Swap(leftChild, rightChild);
//			Utils::Swap(node.rightChild, node.leftChild);
//		}
//
//		if (dist1 == 1e30f)
//		{
//			if (stackPtr == 0)
//			{
//				traceRequest.intersection.Set(rayIndex, intersection);
//				shouldFetchNewRay = true;
//				continue;
//			}
//
//			if (stackPtr == instanceStackDepth)
//			{
//				ray = backupRay;
//
//				mesh.bvh = tlas;
//				instanceStackDepth = INVALID_IDX;
//			}
//			node = mesh.bvh.nodes[stack[--stackPtr]];
//		}
//		else
//		{
//			if (dist2 != 1e30f)
//				stack[stackPtr++] = node.rightChild;
//
//			node = leftChild;
//		}
//	}
//}
//
//inline __device__ void BVH2TraceShadow(D_Mesh* meshes, D_MeshInstance* meshInstances, D_ShadowTraceRequestSOA shadowTraceRequest, int32_t traceSize, int32_t* traceCount, float3* pathRadiance)
//{
//	D_Mesh mesh;
//	NXB::BVH::Node node;
//
//	uint32_t stack[32];
//	uint32_t stackPtr = 0;
//
//	uint32_t instanceStackDepth;
//	uint32_t instanceIdx;
//
//	D_Ray ray, backupRay;
//	int32_t rayIndex;
//	float3 radiance;
//	float hitDistance;
//	uint32_t pixelIdx;
//
//	bool anyHit = true;
//	bool shouldFetchNewRay = true;
//
//	while (true)
//	{
//		if (!anyHit && shouldFetchNewRay)
//			pathRadiance[pixelIdx] += radiance;
//
//		if (shouldFetchNewRay)
//		{
//			shouldFetchNewRay = false;
//
//			rayIndex = atomicAdd(traceCount, 1);
//			if (rayIndex >= traceSize)
//				return;
//
//			mesh.bvh = tlas;
//			node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];
//			ray = shadowTraceRequest.ray.Get(rayIndex);
//			backupRay = ray;
//			hitDistance = shadowTraceRequest.hitDistance[rayIndex];
//			pixelIdx = shadowTraceRequest.pixelIdx[rayIndex];
//			radiance = shadowTraceRequest.radiance[rayIndex];
//			instanceStackDepth = INVALID_IDX;
//			anyHit = false;
//		}
//
//		if (node.leftChild == INVALID_IDX)
//		{
//			// Reached a mesh instance
//			if (instanceStackDepth == INVALID_IDX)
//			{
//				instanceIdx = node.rightChild;
//				instanceStackDepth = stackPtr;
//
//				const D_MeshInstance& meshInstance = meshInstances[instanceIdx];
//				mesh = meshes[meshInstance.meshIdx];
//				node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];
//
//				ray.origin = meshInstance.invTransform.TransformPoint(ray.origin);
//				ray.direction = meshInstance.invTransform.TransformVector(ray.direction);
//				ray.invDirection = 1.0f / ray.direction;
//			}
//			// Reached a primitive
//			else
//			{
//				uint32_t triangleIdx = node.rightChild;
//				NXB::Triangle triangle = mesh.triangles[triangleIdx];
//				if (TriangleTraceShadow(triangle, ray, hitDistance))
//				{
//					stackPtr = 0;
//					anyHit = true;
//					shouldFetchNewRay = true;
//					continue;
//				}
//				if (stackPtr == 0)
//				{
//					shouldFetchNewRay = true;
//					continue;
//				}
//				if (stackPtr == instanceStackDepth)
//				{
//					ray = backupRay;
//
//					mesh.bvh = tlas;
//					instanceStackDepth = INVALID_IDX;
//				}
//				node = mesh.bvh.nodes[stack[--stackPtr]];
//			}
//			continue;
//		}
//
//		NXB::BVH::Node leftChild = mesh.bvh.nodes[node.leftChild];
//		NXB::BVH::Node rightChild = mesh.bvh.nodes[node.rightChild];
//		float dist1 = AABBTrace(ray, hitDistance, leftChild.bounds);
//		float dist2 = AABBTrace(ray, hitDistance, rightChild.bounds);
//
//		if (dist1 > dist2)
//		{
//			Utils::Swap(dist1, dist2);
//			Utils::Swap(leftChild, rightChild);
//			Utils::Swap(node.rightChild, node.leftChild);
//		}
//
//		if (dist1 == 1e30f)
//		{
//			if (stackPtr == 0)
//			{
//				shouldFetchNewRay = true;
//				continue;
//			}
//
//			if (stackPtr == instanceStackDepth)
//			{
//				ray = backupRay;
//
//				mesh.bvh = tlas;
//				instanceStackDepth = INVALID_IDX;
//			}
//			node = mesh.bvh.nodes[stack[--stackPtr]];
//		}
//		else
//		{
//			if (dist2 != 1e30f)
//				stack[stackPtr++] = node.rightChild;
//
//			node = leftChild;
//		}
//	}
//}
//
//
//inline __device__ void BVH2TraceVisualize(D_Mesh* meshes, D_MeshInstance* meshInstances, D_TraceRequestSOA traceRequest, D_PathStateSOA pathState, uint32_t bounce, int32_t traceSize, int32_t* traceCount)
//{
//	D_Mesh mesh;
//	NXB::BVH::Node node;
//	D_Intersection intersection;
//
//	uint32_t stack[32];
//	uint32_t stackPtr = 0;
//
//	uint32_t instanceStackDepth;
//	uint32_t instanceIdx;
//
//	D_Ray ray, backupRay;
//	int32_t rayIndex;
//
//	bool shouldFetchNewRay = true;
//	float boundsHit = 0;
//	float attenuation = 1;
//
//	while (true)
//	{
//		if (shouldFetchNewRay)
//		{
//			boundsHit = 0;
//			shouldFetchNewRay = false;
//
//			rayIndex = atomicAdd(traceCount, 1);
//			if (rayIndex >= traceSize)
//				return;
//
//			mesh.bvh = tlas;
//			node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];
//			ray = traceRequest.ray.Get(rayIndex);
//			backupRay = ray;
//			intersection.hitDistance = 1e30f;
//			instanceStackDepth = INVALID_IDX;
//		}
//
//		if (node.leftChild == INVALID_IDX)
//		{
//			// Reached a mesh instance
//			if (instanceStackDepth == INVALID_IDX)
//			{
//				instanceIdx = node.rightChild;
//				instanceStackDepth = stackPtr;
//
//				const D_MeshInstance& meshInstance = meshInstances[instanceIdx];
//				mesh = meshes[meshInstance.meshIdx];
//				node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];
//
//				ray.origin = meshInstance.invTransform.TransformPoint(ray.origin);
//				ray.direction = meshInstance.invTransform.TransformVector(ray.direction);
//				ray.invDirection = 1.0f / ray.direction;
//			}
//			// Reached a primitive
//			else
//			{
//				uint32_t triangleIdx = node.rightChild;
//				NXB::Triangle triangle = mesh.triangles[triangleIdx];
//				TriangleTrace(triangle, ray, intersection, instanceIdx, triangleIdx);
//				if (stackPtr == 0)
//				{
//					intersection.hitDistance = 1.0e30f;
//					traceRequest.intersection.Set(rayIndex, intersection);
//					pathState.radiance[traceRequest.pixelIdx[rayIndex]] = make_float3(0.0f, boundsHit / 50.0f, max(0.0f, 1.0f - boundsHit / 50.0f));
//					shouldFetchNewRay = true;
//					continue;
//				}
//				if (stackPtr == instanceStackDepth)
//				{
//					ray = backupRay;
//
//					mesh.bvh = tlas;
//					instanceStackDepth = INVALID_IDX;
//				}
//				node = mesh.bvh.nodes[stack[--stackPtr]];
//			}
//			continue;
//		}
//
//		NXB::BVH::Node leftChild = mesh.bvh.nodes[node.leftChild];
//		NXB::BVH::Node rightChild = mesh.bvh.nodes[node.rightChild];
//		float dist1 = AABBTrace(ray, intersection.hitDistance, leftChild.bounds);
//		float dist2 = AABBTrace(ray, intersection.hitDistance, rightChild.bounds);
//
//		if (dist1 > dist2)
//		{
//			Utils::Swap(dist1, dist2);
//			Utils::Swap(leftChild, rightChild);
//			Utils::Swap(node.rightChild, node.leftChild);
//		}
//
//		if (dist1 == 1e30f)
//		{
//			if (stackPtr == 0)
//			{
//				intersection.hitDistance = 1.0e30f;
//				traceRequest.intersection.Set(rayIndex, intersection);
//				pathState.radiance[traceRequest.pixelIdx[rayIndex]] = make_float3(0.0f, boundsHit / 50.0f, max(0.0f, 1.0f - boundsHit / 50.0f));
//				shouldFetchNewRay = true;
//				continue;
//			}
//
//			if (stackPtr == instanceStackDepth)
//			{
//				ray = backupRay;
//
//				mesh.bvh = tlas;
//				instanceStackDepth = INVALID_IDX;
//			}
//			node = mesh.bvh.nodes[stack[--stackPtr]];
//		}
//		else
//		{
//			if (stackPtr >= instanceStackDepth)
//				boundsHit += 1;
//			if (dist2 != 1e30f)
//			{
//				if (stackPtr >= instanceStackDepth)
//					boundsHit += 1;
//				stack[stackPtr++] = node.rightChild;
//			}
//
//			node = leftChild;
//		}
//	}
//}
//
//
//inline __device__ void BVH2TraceVisualizeWireframe(D_Mesh* meshes, D_MeshInstance* meshInstances, D_TraceRequestSOA traceRequest, D_PathStateSOA pathState, uint32_t bounce, int32_t traceSize, int32_t* traceCount)
//{
//	D_Mesh mesh;
//	NXB::BVH::Node node;
//	D_Intersection intersection;
//
//	uint32_t stack[32];
//	uint32_t stackPtr = 0;
//
//	uint32_t instanceStackDepth;
//	uint32_t instanceIdx;
//
//	D_Ray ray, backupRay;
//	int32_t rayIndex;
//
//	bool shouldFetchNewRay = true;
//	float boundsHit = 0;
//	float attenuation = 1;
//	bool hitEdge = false;
//	float3 instanceScale = make_float3(1.0f);
//
//	while (true)
//	{
//		if (shouldFetchNewRay)
//		{
//			instanceScale = make_float3(1.0f);
//			boundsHit = 0;
//			hitEdge = false;
//			shouldFetchNewRay = false;
//
//			rayIndex = atomicAdd(traceCount, 1);
//			if (rayIndex >= traceSize)
//				return;
//
//			mesh.bvh = tlas;
//			node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];
//			ray = traceRequest.ray.Get(rayIndex);
//			backupRay = ray;
//			intersection.hitDistance = 1e30f;
//			instanceStackDepth = INVALID_IDX;
//		}
//
//		if (node.leftChild == INVALID_IDX)
//		{
//			// Reached a mesh instance
//			if (instanceStackDepth == INVALID_IDX)
//			{
//				instanceIdx = node.rightChild;
//				instanceStackDepth = stackPtr;
//
//				const D_MeshInstance& meshInstance = meshInstances[instanceIdx];
//				mesh = meshes[meshInstance.meshIdx];
//				node = mesh.bvh.nodes[mesh.bvh.nodeCount - 1];
//
//				ray.origin = meshInstance.invTransform.TransformPoint(ray.origin);
//				instanceScale = meshInstance.transform.GetScale();
//				ray.direction = meshInstance.invTransform.TransformVector(ray.direction);
//				ray.invDirection = 1.0f / ray.direction;
//			}
//			// Reached a primitive
//			else
//			{
//				if (stackPtr == 0)
//				{
//					traceRequest.intersection.Set(rayIndex, intersection);
//					pathState.radiance[traceRequest.pixelIdx[rayIndex]] = make_float3(1.0f, (float)!hitEdge, (float)!hitEdge);
//					shouldFetchNewRay = true;
//					continue;
//				}
//				if (stackPtr == instanceStackDepth)
//				{
//					ray = backupRay;
//					instanceScale = make_float3(1.0f);
//
//					mesh.bvh = tlas;
//					instanceStackDepth = INVALID_IDX;
//				}
//				node = mesh.bvh.nodes[stack[--stackPtr]];
//			}
//			continue;
//		}
//
//		NXB::BVH::Node leftChild = mesh.bvh.nodes[node.leftChild];
//		NXB::BVH::Node rightChild = mesh.bvh.nodes[node.rightChild];
//		float dist1 = AABBTraceWireframe(ray, intersection.hitDistance, leftChild.bounds, hitEdge, instanceScale);
//		float dist2 = AABBTraceWireframe(ray, intersection.hitDistance, rightChild.bounds, hitEdge, instanceScale);
//
//		if (dist1 > dist2)
//		{
//			Utils::Swap(dist1, dist2);
//			Utils::Swap(leftChild, rightChild);
//			Utils::Swap(node.rightChild, node.leftChild);
//		}
//
//		if (dist1 == 1e30f)
//		{
//			if (stackPtr == 0)
//			{
//				traceRequest.intersection.Set(rayIndex, intersection);
//				pathState.radiance[traceRequest.pixelIdx[rayIndex]] = make_float3(1.0f, (float)!hitEdge, (float)!hitEdge);
//				//pathState.radiance[traceRequest.pixelIdx[rayIndex]] = make_float3(0.0f, boundsHit / 200.0f, max(0.0f, 1.0f - boundsHit / 200.0f));
//				shouldFetchNewRay = true;
//				continue;
//			}
//
//			if (stackPtr == instanceStackDepth)
//			{
//				ray = backupRay;
//				instanceScale = make_float3(1.0f);
//
//				mesh.bvh = tlas;
//				instanceStackDepth = INVALID_IDX;
//			}
//			node = mesh.bvh.nodes[stack[--stackPtr]];
//		}
//		else
//		{
//			if (stackPtr >= instanceStackDepth)
//				boundsHit += 1;
//			if (dist2 != 1e30f)
//			{
//				if (stackPtr >= instanceStackDepth)
//					boundsHit += 1;
//				stack[stackPtr++] = node.rightChild;
//			}
//
//			node = leftChild;
//		}
//	}
//}
