#pragma once

#include <cuda_runtime_api.h>

#include "Math/Mat4.h"
#include "Geometry/AABB.h"
#include "Cuda/BVH/BVHInstance.cuh"
#include "Geometry/BVH/BVH8.h"

class BVHInstance
{
public:
	BVHInstance() = default;
	BVHInstance(unsigned int blasIdx, BVH8* bvhs)
		: m_BvhIdx(blasIdx), m_Bvh(bvhs) 
	{
		Mat4 m;
		SetTransform(m);
	}

	void SetTransform(Mat4& t);
	void SetTransform(float3 pos, float3 r, float3 s);

	const AABB& GetBounds() const { return m_Bounds; }

	void AssignMaterial(int mIdx);

	static D_BVHInstance ToDevice(const BVHInstance& bvhInstance);

private:
	unsigned int m_BvhIdx = 0;
	BVH8* m_Bvh;
	Mat4 m_InvTransform;
	Mat4 m_Transform;
	AABB m_Bounds;
	int m_MaterialId;
};
