#pragma once
#include <vector>
#include "Utils/Utils.h"
#include "Geometry/AABB.h"
#include "Geometry/Ray.h"
#include "Geometry/Triangle.h"
#include "Cuda/Geometry/Triangle.cuh"
#include "Cuda/BVH/BVH2.cuh"

// Standard SAH-Based BVH with binned building adapted from Jacco Bikker's guides
// See https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
// The main difference is that the leaves contain only one primitive to allow
// for collapsing the nodes to construct more advanced BVHs (e.g. compressed wide BVHs)

#define BINS 8

struct BVH2Node
{
	// Bounds
	float3 aabbMin, aabbMax;

	// Either the index of the left child node, or the first index in the triangles list
	union {
		uint32_t leftNode;
		uint32_t firstTriIdx;
	};

	// Number of triangles in the node (0 if not leaf)
	uint32_t triCount;

	inline bool IsLeaf() const { return triCount > 0; }
	inline float Cost() const
	{
		float3 diag = aabbMax - aabbMin;
		return (diag.x * diag.y + diag.y * diag.z + diag.x * diag.z) * triCount;
	}
};

typedef struct Bin 
{
	AABB bounds;
	int triCount = 0;
} Bin;

class BVH2
{
public:
	BVH2() = default;
	BVH2(const std::vector<Triangle>& triangles);

	void Build();

private:
	void ComputeTrianglesAABB();
	void SplitNodeInHalf(BVH2Node& node);
	void Subdivide(uint32_t nodeIdx);
	void UpdateNodeBounds(uint32_t nodeIdx);
	float FindBestSplitPlane(const BVH2Node& node, int& axis, double& splitPos);

public:
	std::vector<Triangle> triangles;
	std::vector<uint32_t> triangleIdx;
	std::vector<BVH2Node> nodes;
	std::vector<AABB> trianglesAABB;
};


