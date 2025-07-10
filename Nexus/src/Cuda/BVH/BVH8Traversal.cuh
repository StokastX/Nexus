#pragma once

#include <cuda_runtime_api.h>
#include "Utils/Utils.h"
#include "Geometry/BVH/BVH.h"
#include "Cuda/Scene/MeshInstance.cuh"
#include "Cuda/Scene/Mesh.cuh"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Cuda/Geometry/Ray.cuh"
#include "Cuda/Utils.cuh"

// If the ratio of active threads in a warp is less than POSTPONE_RATIO_THRESHOLD, postpone triangle intersection
#define POSTPONE_RATIO_THRESHOLD 0.2

// Constants for dynamic ray fetching
#define N_D 4
#define N_W 16

#define TRAVERSAL_STACK_SIZE 32
#define SHARED_STACK_SIZE 8

__device__ __constant__ NXB::BVH tlas;

inline __device__ uint32_t Octant(const float3& a)
{
	return ((a.x < 0 ? 1 : 0) << 2) | ((a.y < 0 ? 1 : 0) << 1) | ((a.z < 0 ? 1 : 0));
}

// Pop from shared or local stack
inline __device__ uint2 StackPop(
	const uint2 sharedStack[],
	const uint2 localStack[], char& stackPtr
) {
	stackPtr--;
	if (stackPtr < SHARED_STACK_SIZE)
		return sharedStack[threadIdx.x * SHARED_STACK_SIZE + stackPtr];
	else
		return localStack[stackPtr - SHARED_STACK_SIZE];
}

// Push to shared or local stack
inline __device__ void StackPush(
	uint2 sharedStack[],
	uint2 localStack[], char& stackPtr, const uint2& stackEntry
) {
	if (stackPtr < SHARED_STACK_SIZE)
		sharedStack[threadIdx.x * SHARED_STACK_SIZE + stackPtr] = stackEntry;
	else
		localStack[stackPtr - SHARED_STACK_SIZE] = stackEntry;

	stackPtr++;
}


__forceinline__ __device__ void ChildTrace(
	const NXB::BVH::Node* nodes,
	const uint32_t nodeIdx,
	const D_Ray& ray,
	const uint32_t invOctant4,
	const float hitDistance,
	uint2& internalEntry,
	uint2& triangleEntry
)
{
	const float4 p_e_imask				= __ldg(&nodes[nodeIdx].p_e_imask);
	const float4 childidx_tridx_meta	= __ldg(&nodes[nodeIdx].childidx_tridx_meta);
	const float4 qlox_qloy				= __ldg(&nodes[nodeIdx].qlox_qloy);
	const float4 qloz_qhix				= __ldg(&nodes[nodeIdx].qloz_qhix);
	const float4 qhiy_qhiz				= __ldg(&nodes[nodeIdx].qhiy_qhiz);

	const float3 p = make_float3(p_e_imask);
	const uint32_t e_imask = __float_as_uint(p_e_imask.w);

	const byte ex = ExtractByte(e_imask, 0);
	const byte ey = ExtractByte(e_imask, 1);
	const byte ez = ExtractByte(e_imask, 2);

	// We can easily compute 2^ei by shifting ei to the 8 exponent bits and interpreting the result as a float
	float3 transformedDirection = make_float3(
		__uint_as_float(ex << 23) * ray.invDirection.x,
		__uint_as_float(ey << 23) * ray.invDirection.y,
		__uint_as_float(ez << 23) * ray.invDirection.z
	);

	uint32_t hitMask = 0;

	const float3 transformedOrigin = (p - ray.origin) * ray.invDirection;

	#pragma unroll
	for (int i = 0; i < 2; i++)
	{
		const uint32_t meta4 = __float_as_uint(i == 0 ? childidx_tridx_meta.z : childidx_tridx_meta.w);
		const uint32_t isInner4 = (meta4 & (meta4 << 1)) & 0x10101010;
		const uint32_t innerMask4 = SignExtendS8x4(isInner4 << 3);
		const uint32_t bitIndex4 = (meta4 ^ (invOctant4 & innerMask4)) & 0x1f1f1f1f;
		const uint32_t childBits4 = (meta4 >> 5) & 0x07070707;

		const uint32_t qlox = __float_as_uint(i == 0 ? qlox_qloy.x : qlox_qloy.y);
		const uint32_t qhix = __float_as_uint(i == 0 ? qloz_qhix.z : qloz_qhix.w);

		const uint32_t qloy = __float_as_uint(i == 0 ? qlox_qloy.z : qlox_qloy.w);
		const uint32_t qhiy = __float_as_uint(i == 0 ? qhiy_qhiz.x : qhiy_qhiz.y);

		const uint32_t qloz = __float_as_uint(i == 0 ? qloz_qhix.x : qloz_qhix.y);
		const uint32_t qhiz = __float_as_uint(i == 0 ? qhiy_qhiz.z : qhiy_qhiz.w);

		const uint32_t xMin = ray.direction.x < 0.0f ? qhix : qlox;
		const uint32_t xMax = ray.direction.x < 0.0f ? qlox : qhix;

		const uint32_t yMin = ray.direction.y < 0.0f ? qhiy : qloy;
		const uint32_t yMax = ray.direction.y < 0.0f ? qloy : qhiy;

		const uint32_t zMin = ray.direction.z < 0.0f ? qhiz : qloz;
		const uint32_t zMax = ray.direction.z < 0.0f ? qloz : qhiz;

		#pragma unroll
		for (int j = 0; j < 4; j++)
		{
			// Extract j-th byte
			float3 tmin3 = make_float3(float(ExtractByte(xMin, j)), float(ExtractByte(yMin, j)), float(ExtractByte(zMin, j)));
			float3 tmax3 = make_float3(float(ExtractByte(xMax, j)), float(ExtractByte(yMax, j)), float(ExtractByte(zMax, j)));

			// Account for grid origin and scale
			tmin3 = tmin3 * transformedDirection + transformedOrigin;
			tmax3 = tmax3 * transformedDirection + transformedOrigin;

			const float tmin = vMaxMax(tmin3.x, tmin3.y, fmaxf(tmin3.z, 0.0f));
			const float tmax = vMinMin(tmax3.x, tmax3.y, fminf(tmax3.z, hitDistance));

			const bool intersected = tmin <= tmax;
			if (intersected) {
				const uint32_t childBits = ExtractByte(childBits4, j);
				const uint32_t bitIndex  = ExtractByte(bitIndex4,  j);

				hitMask |= childBits << bitIndex;
			}
		}
	}
	const uint32_t imask = ExtractByte(__float_as_uint(p_e_imask.w), 3);

	internalEntry.x = __float_as_uint(childidx_tridx_meta.x);
	internalEntry.y = (hitMask & 0xff000000) | imask;

	triangleEntry.x = __float_as_uint(childidx_tridx_meta.y);
	triangleEntry.y = (hitMask & 0x00ffffff);
}

inline __device__ void BVH8Trace(D_Mesh* meshes, D_MeshInstance* meshInstances, D_TraceRequestSOA traceRequest, int32_t traceSize, int32_t* traceCount)
{
	__shared__ uint2 sharedStack[BLOCK_SIZE * SHARED_STACK_SIZE];
	uint2 stack[TRAVERSAL_STACK_SIZE - SHARED_STACK_SIZE];
	char stackPtr = 0;

	D_Ray ray, backupRay;
	int32_t rayIndex;
	D_Intersection intersection;

	uint32_t instanceIdx;
	char instanceStackDepth;
	NXB::BVH bvh;
	D_Mesh mesh;

	uint2 nodeEntry;
	uint2 triangleEntry;
	uint32_t invOctant, invOctant4;

	NXB::BVH::Node* nodes;

	unsigned char lostWork;
	bool shouldFetchNewRay = true;

	while (true)
	{
		if (shouldFetchNewRay)
		{
			shouldFetchNewRay = false;
			lostWork = 0;

			rayIndex = atomicAdd(traceCount, 1);
			if (rayIndex >= traceSize)
				return;

			bvh = tlas;
			nodes = tlas.nodes;
			ray = traceRequest.ray.Get(rayIndex);
			backupRay = ray;
			invOctant = 7 - Octant(ray.direction);
			invOctant4 = invOctant * 0x01010101;
			intersection.hitDistance = 1e30f;
			instanceStackDepth = -1;

			nodeEntry = make_uint2(0, 0x80000000);
		}

		while (true)
		{

			// If the hits field is different from 0, it is an internal node entry
			if (nodeEntry.y & 0xff000000)
			{
				// Position of the first non zero bit
				const int nodeOffset = 31 - __clz(nodeEntry.y);

				// Set the hits bit of the selected node to 0
				nodeEntry.y &= ~(1 << nodeOffset);

				// If some nodes are remaining in the hits field
				if (nodeEntry.y & 0xff000000)
				{
					StackPush(sharedStack, stack, stackPtr, nodeEntry);
				};

				// Slot in (0 .. 7) referring to the octant order in which the node should be traversed
				const int nodeSlot = (nodeOffset - 24) ^ invOctant;

				// We need to account for the number of internal nodes in the parent node. The relative
				// index is thus the number of neighboring internal nodes stored in the lower child slots
				const int relativeNodeIdx = __popc(nodeEntry.y & ~(0xffffffff << nodeSlot));

				assert(nodeEntry.x + relativeNodeIdx < bvh.nodeCount);

				ChildTrace(nodes, nodeEntry.x + relativeNodeIdx, ray, invOctant4, intersection.hitDistance, nodeEntry, triangleEntry);
			}
			else
			{
				triangleEntry = nodeEntry;
				nodeEntry = make_uint2(0);
			}

			const float postponeThreshold = __popc(__activemask()) * POSTPONE_RATIO_THRESHOLD;

			while (triangleEntry.y)
			{
				// We reached a TLAS leaf == BVHInstance
				if (instanceStackDepth == -1)
				{
					const int triangleOffset = 31 - __clz(triangleEntry.y);

					// Set the hits bit of the selected triangle to 0
					triangleEntry.y &= ~(1 << (triangleOffset));

					instanceIdx = tlas.primIdx[triangleEntry.x + triangleOffset];

					// If some leaf entries are remaining in TLAS
					if (triangleEntry.y)
						StackPush(sharedStack, stack, stackPtr, triangleEntry);

					// If some child nodes are remaining in TLAS node entry
					if (nodeEntry.y & 0xff000000)
						StackPush(sharedStack, stack, stackPtr, nodeEntry);

					instanceStackDepth = stackPtr;

					const D_MeshInstance& bvhInstance = meshInstances[instanceIdx];
					mesh = meshes[bvhInstance.meshIdx];
					bvh = mesh.bvh;
					nodes = bvh.nodes;

					nodeEntry = make_uint2(0, 0x80000000);

					invOctant = 7 - Octant(ray.direction);
					invOctant4 = invOctant * 0x01010101;

					ray.origin = bvhInstance.invTransform.TransformPoint(ray.origin);
					ray.direction = bvhInstance.invTransform.TransformVector(ray.direction);
					ray.invDirection = 1.0f / ray.direction;

					break;
				}

				float ratio = __popc(__activemask());

				// If the ratio of active threads in the warp performing triangle
				// intersection is less than the threshold, postpone
				if (ratio < postponeThreshold)
				{
					StackPush(sharedStack, stack, stackPtr, triangleEntry);
					break;
				}

				const int triangleOffset = 31 - __clz(triangleEntry.y);

				// Set the hits bit of the selected triangle to 0
				triangleEntry.y &= ~(1 << (triangleOffset));

				// Fetch the triangle index
				const uint32_t triangleIdx = bvh.primIdx[triangleEntry.x + triangleOffset];

				assert(triangleIdx < bvh.primCount);

				// Ray triangle intersection
				NXB::Triangle triangle = mesh.triangles[triangleIdx];
				TriangleTrace(triangle, ray, intersection, instanceIdx, triangleIdx);
			}

			// If the node entry is empty (hits field equals 0), pop from the stack
			if ((nodeEntry.y & 0xff000000) == 0)
			{
				if (stackPtr == 0)
				{
					traceRequest.intersection.Set(rayIndex, intersection);
					shouldFetchNewRay = true;
					break;
				}

				if (stackPtr == instanceStackDepth)
				{
					ray = backupRay;

					invOctant = 7 - Octant(ray.direction);
					invOctant4 = invOctant * 0x01010101;
					bvh = tlas;
					nodes = tlas.nodes;

					instanceStackDepth = -1;
				}
				nodeEntry = StackPop(sharedStack, stack, stackPtr);
			}

			lostWork += WARP_SIZE - __popc(__activemask()) - N_D;
			if (lostWork >= N_W)
				break;
		}
	}
}


// Shadow ray tracing: true if any hit
inline __device__ void BVH8TraceShadow(D_Mesh* meshes, D_MeshInstance* meshInstances, D_ShadowTraceRequestSOA shadowTraceRequest, int32_t traceSize, int32_t* traceCount, float3* pathRadiance)
{
	__shared__ uint2 sharedStack[BLOCK_SIZE * SHARED_STACK_SIZE];
	uint2 stack[TRAVERSAL_STACK_SIZE - SHARED_STACK_SIZE];
	char stackPtr = 0;

	D_Ray ray, backupRay;
	int32_t rayIndex;
	uint32_t pixelIdx;
	float3 radiance;
	float hitDistance;
	bool anyHit;

	uint32_t instanceIdx;
	char instanceStackDepth;
	NXB::BVH bvh;
	D_Mesh mesh;

	uint2 nodeEntry;
	uint2 triangleEntry;
	uint32_t invOctant, invOctant4;

	NXB::BVH::Node* nodes;

	unsigned char lostWork;
	bool shouldFetchNewRay = true;

	while (true)
	{
		if (shouldFetchNewRay)
		{
			shouldFetchNewRay = false;
			lostWork = 0;

			rayIndex = atomicAdd(traceCount, 1);
			if (rayIndex >= traceSize)
				return;

			bvh = tlas;
			nodes = tlas.nodes;
			ray = shadowTraceRequest.ray.Get(rayIndex);
			backupRay = ray;
			invOctant = 7 - Octant(ray.direction);
			invOctant4 = invOctant * 0x01010101;
			hitDistance = shadowTraceRequest.hitDistance[rayIndex];
			pixelIdx = shadowTraceRequest.pixelIdx[rayIndex];
			radiance = shadowTraceRequest.radiance[rayIndex];
			instanceStackDepth = -1;
			anyHit = false;

			nodeEntry = make_uint2(0, 0x80000000);
		}

		while (true)
		{
			// If the hits field is different from 0, it is an internal node entry
			if (nodeEntry.y & 0xff000000)
			{
				// Position of the first non zero bit
				const int nodeOffset = 31 - __clz(nodeEntry.y);

				// Set the hits bit of the selected node to 0
				nodeEntry.y &= ~(1 << nodeOffset);

				// If some nodes are remaining in the hits field
				if (nodeEntry.y & 0xff000000)
				{
					StackPush(sharedStack, stack, stackPtr, nodeEntry);
				};

				// Slot in (0 .. 7) referring to the octant order in which the node should be traversed
				const int nodeSlot = (nodeOffset - 24) ^ invOctant;

				// We need to account for the number of internal nodes in the parent node. The relative
				// index is thus the number of neighboring internal nodes stored in the lower child slots
				const int relativeNodeIdx = __popc(nodeEntry.y & ~(0xffffffff << nodeSlot));

				assert(nodeEntry.x + relativeNodeIdx < bvh.nodeCount);

				ChildTrace(nodes, nodeEntry.x + relativeNodeIdx, ray, invOctant4, hitDistance, nodeEntry, triangleEntry);
			}
			else
			{
				triangleEntry = nodeEntry;
				nodeEntry = make_uint2(0);
			}

			const float postponeThreshold = __popc(__activemask()) * POSTPONE_RATIO_THRESHOLD;

			while (triangleEntry.y)
			{
				// We reached a TLAS leaf == BVHInstance
				if (instanceStackDepth == -1)
				{
					const int triangleOffset = 31 - __clz(triangleEntry.y);

					// Set the hits bit of the selected triangle to 0
					triangleEntry.y &= ~(1 << (triangleOffset));

					instanceIdx = tlas.primIdx[triangleEntry.x + triangleOffset];

					// If some leaf entries are remaining in TLAS
					if (triangleEntry.y)
						StackPush(sharedStack, stack, stackPtr, triangleEntry);

					// If some child nodes are remaining in TLAS node entry
					if (nodeEntry.y & 0xff000000)
						StackPush(sharedStack, stack, stackPtr, nodeEntry);

					instanceStackDepth = stackPtr;

					const D_MeshInstance& bvhInstance = meshInstances[instanceIdx];
					mesh = meshes[bvhInstance.meshIdx];
					bvh = mesh.bvh;
					nodes = bvh.nodes;

					nodeEntry = make_uint2(0, 0x80000000);

					invOctant = 7 - Octant(ray.direction);
					invOctant4 = invOctant * 0x01010101;

					ray.origin = bvhInstance.invTransform.TransformPoint(ray.origin);
					ray.direction = bvhInstance.invTransform.TransformVector(ray.direction);
					ray.invDirection = 1.0f / ray.direction;

					break;
				}

				float ratio = __popc(__activemask());

				// If the ratio of active threads in the warp performing triangle
				// intersection is less than the threshold, postpone
				if (ratio < postponeThreshold)
				{
					StackPush(sharedStack, stack, stackPtr, triangleEntry);
					break;
				}

				const int triangleOffset = 31 - __clz(triangleEntry.y);

				// Set the hits bit of the selected triangle to 0
				triangleEntry.y &= ~(1 << (triangleOffset));

				// Fetch the triangle index
				const uint32_t triangleIdx = bvh.primIdx[triangleEntry.x + triangleOffset];
				NXB::Triangle triangle = mesh.triangles[triangleIdx];

				assert(triangleIdx < bvh.primCount);

				// Ray triangle intersection
				if (TriangleTraceShadow(triangle, ray, hitDistance))
				{
					anyHit = true;
					break;
				}
			}

			if (anyHit)
			{
				stackPtr = 0;
				shouldFetchNewRay = true;
				break;
			}

			// If the node entry is empty (hits field equals 0), pop from the stack
			if ((nodeEntry.y & 0xff000000) == 0)
			{
				if (stackPtr == 0)
				{
					shouldFetchNewRay = true;
					break;
				}

				if (stackPtr == instanceStackDepth)
				{
					ray = backupRay;

					invOctant = 7 - Octant(ray.direction);
					invOctant4 = invOctant * 0x01010101;
					bvh = tlas;
					nodes = tlas.nodes;

					instanceStackDepth = -1;
				}

				nodeEntry = StackPop(sharedStack, stack, stackPtr);
			}

			lostWork += WARP_SIZE - __popc(__activemask()) - N_D;
			if (lostWork >= N_W)
				break;
		}
		if (!anyHit && shouldFetchNewRay)
			pathRadiance[pixelIdx] += radiance;
	}
}
