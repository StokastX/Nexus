#include <vector>
#include <numeric>
#include "BVH2.h"
#include "Geometry/AABB.h"
#include "Utils/Utils.h"

BVH2::BVH2(const std::vector<Triangle>& tri)
{
	triangles = tri;
	triangleIdx = std::vector<uint32_t>(tri.size());
}

void BVH2::Build()
{
	// Fill triangle indices with consecutive integers starting from 0
	std::iota(triangleIdx.begin(), triangleIdx.end(), 0);

	BVH2Node root;
	root.leftNode = 0;
	root.triCount = triangles.size();
	nodes.push_back(root);

	ComputeTrianglesAABB();
	UpdateNodeBounds(0);
	Subdivide(0);
}

void BVH2::ComputeTrianglesAABB()
{
	for (const Triangle& triangle: triangles)
	{
		AABB triangleAABB;
		triangleAABB.Grow(triangle.pos0);
		triangleAABB.Grow(triangle.pos1);
		triangleAABB.Grow(triangle.pos2);
		trianglesAABB.push_back(triangleAABB);
	}
}

void BVH2::SplitNodeInHalf(BVH2Node& node)
{
	int leftChildIdx = nodes.size();
	BVH2Node leftChild;
	leftChild.firstTriIdx = node.firstTriIdx;
	leftChild.triCount = node.triCount / 2;

	int rightChildIdx = nodes.size() + 1;
	BVH2Node rightChild;
	rightChild.firstTriIdx = node.firstTriIdx + node.triCount / 2;
	rightChild.triCount = node.triCount - node.triCount / 2;
	node.leftNode = leftChildIdx;
	node.triCount = 0;

	// node should not be used after this since its reference will be invalidated
	nodes.push_back(leftChild);
	nodes.push_back(rightChild);

	UpdateNodeBounds(leftChildIdx);
	UpdateNodeBounds(rightChildIdx);

	Subdivide(leftChildIdx);
	Subdivide(rightChildIdx);
}

void BVH2::Subdivide(uint32_t nodeIdx)
{
	BVH2Node& node = nodes[nodeIdx];

	int axis = -1;
	double splitPos;
	float splitCost = FindBestSplitPlane(node, axis, splitPos);
	float nodeCost = node.Cost();

	// Set every leaf primitive count to 1 to allow for collapsing nodes for constructing other BVHs
	if (node.triCount == 1)
		return;

	// If in one node triangles have the same centroid, they cannot be 
	// separated by chopped binning. We have to separate them manually
	else if (axis == -1)
	{
		SplitNodeInHalf(node);
		return;
	}

	// Normally, we would return if the split cost is greater than the parent node cost
	//if (splitCost > nodeCost)
	//	return;

	int i = node.firstTriIdx;
	int j = i + node.triCount - 1;

	while (i <= j)
	{
		float centroidCoord = *((float*)&triangles[triangleIdx[i]].centroid + axis);
		if (centroidCoord < splitPos)
			i++;
		else
			Utils::Swap(triangleIdx[i], triangleIdx[j--]);
	}
	
	int leftCount = i - node.firstTriIdx;
	if (leftCount == 0 || leftCount == node.triCount)
	{
		if (node.triCount == 1)
			return;

		SplitNodeInHalf(node);
		return;
	}

	int leftChildIdx = nodes.size();
	int rightChildIdx = leftChildIdx + 1;

	BVH2Node leftChild;
	leftChild.firstTriIdx = node.firstTriIdx;
	leftChild.triCount = leftCount;

	BVH2Node rightChild;
	rightChild.firstTriIdx = i;
	rightChild.triCount = node.triCount - leftCount;

	node.leftNode = leftChildIdx;
	node.triCount = 0;

	nodes.push_back(leftChild);
	nodes.push_back(rightChild);

	UpdateNodeBounds(leftChildIdx);
	UpdateNodeBounds(rightChildIdx);

	Subdivide(leftChildIdx);
	Subdivide(rightChildIdx);
}

void BVH2::UpdateNodeBounds(uint32_t nodeIdx)
{
	BVH2Node& node = nodes[nodeIdx];
	node.aabbMin = make_float3(1e30f);
	node.aabbMax = make_float3(-1e30f);
	for (uint32_t first = node.firstTriIdx, i = 0; i < node.triCount; i++)
	{
		uint32_t leafTriIdx = triangleIdx[first + i];
		const AABB& triangleAABB = trianglesAABB[leafTriIdx];
		node.aabbMin = fminf(node.aabbMin, triangleAABB.bMin);
		node.aabbMax = fmaxf(node.aabbMax, triangleAABB.bMax);
	}
}

float BVH2::FindBestSplitPlane(const BVH2Node& node, int& axis, double& splitPos)
{
	float bestCost = 1e30f;
	for (int a = 0; a < 3; a++)
	{
		float boundsMin = 1e30f, boundsMax = -1e30f;
		for (uint32_t i = 0; i < node.triCount; i++)
		{
			Triangle& triangle = triangles[triangleIdx[node.firstTriIdx + i]];
			boundsMin = fmin(boundsMin, *((float*)&triangle.centroid + a));
			boundsMax = fmax(boundsMax, *((float*)&triangle.centroid + a));
		}
		if (boundsMin == boundsMax)
			continue;

		struct Bin { AABB bounds; int triCount = 0; } bins[BINS];
		double scale = BINS / (boundsMax - boundsMin);

		for (uint32_t i = 0; i < node.triCount; i++)
		{
			Triangle& triangle = triangles[triangleIdx[node.firstTriIdx + i]];
			const AABB triangleAABB = trianglesAABB[triangleIdx[node.firstTriIdx + i]];
			float centroidCoord = *((float*)&triangle.centroid + a);
			int binIdx = min((int)(BINS - 1), (int)((centroidCoord - boundsMin) * scale));
			bins[binIdx].triCount++;
			bins[binIdx].bounds.bMin = fminf(bins[binIdx].bounds.bMin, triangleAABB.bMin);
			bins[binIdx].bounds.bMax = fmaxf(bins[binIdx].bounds.bMax, triangleAABB.bMax);
		}

		float leftArea[BINS - 1], rightArea[BINS - 1];
		int leftCount[BINS - 1], rightCount[BINS - 1];
		AABB leftBox, rightBox;
		int leftSum = 0, rightSum = 0;

		for (int i = 0; i < BINS - 1; i++)
		{
			leftSum += bins[i].triCount;
			leftCount[i] = leftSum;
			leftBox.Grow(bins[i].bounds);
			leftArea[i] = leftBox.Area();

			rightSum += bins[BINS - 1 - i].triCount;
			rightCount[BINS - 2 - i] = rightSum;
			rightBox.Grow(bins[BINS - 1 - i].bounds);
			rightArea[BINS - 2 - i] = rightBox.Area();
		}

		scale = (boundsMax - boundsMin) / BINS;
		for (int i = 0; i < BINS - 1; i++)
		{
			float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
			if (planeCost < bestCost)
			{
				axis = a;
				splitPos = boundsMin + scale * (i + 1);
				bestCost = planeCost;
			}
		}
	}
	return bestCost;
}


