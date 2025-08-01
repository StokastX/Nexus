#pragma once

#include <iostream>
#include <vector>
#include "BVH.h"

#define C_PRIM 0.3f  // Cost of a ray-primitive intersection
#define C_NODE 1.0f  // Cost of a ray-node intersection
#define P_MAX  3	 // Maximum allowed leaf size
#define N_Q 8		 // Number of bits used to store the childs' AABB coordinates

class BVH8Builder
{
public:

	BVH8Builder(const NXB::BVH2& bvh2);

	enum struct Decision
	{
		UNDEFINED = -1,
		LEAF,
		INTERNAL,
		DISTRIBUTE
	};

	struct NodeEval
	{
		// SAH cost of node n at index i
		float cost;

		// Decision made for the node
		Decision decision = Decision::UNDEFINED;

		// Left and right count if decision is DISTRIBUTE
		int leftCount, rightCount;
	};

	NXB::BVH8 Build();
	int ComputeNodeTriCount(int nodeIdx, int triBaseIdx);
	float ComputeNodeCost(uint32_t nodeIdx, int i);
	void Init();

	void CollapseNode(NXB::BVH8& bvh8, uint32_t nodeIdxBvh2, uint32_t nodeIdxBvh8);

private:

	// Cleaf(n)
	inline float CLeaf(const NXB::BVH2::Node& node, int triCount);

	// Cinternal(n)
	float CInternal(const NXB::BVH2::Node& node, int& leftCount, int& rightCount);

	// Cdistribute(n, j)
	float CDistribute(const NXB::BVH2::Node& node, int j, int& leftCount, int& rightCount);

	// Returns the indices of the node's children
	void GetChildrenIndices(uint32_t nodeIdxBvh2, int *indices, int i, int& indicesCount);

	int CountTriangles(NXB::BVH8& bvh8, uint32_t nodeIdxBvh2);

	// Order the children in a given node
	void OrderChildren(uint32_t nodeIdxBvh2, int* childrenIndices);

private:
	NXB::BVH2 m_Bvh2;

	// Optimal SAH cost C(n, i) with decisions
	std::vector<std::vector<NodeEval>> m_Evals;

	// Number of triangles in the subtree of the node i
	std::vector<int> m_TriCount;

	// Base triangle index of the subtree of the node i
	std::vector<int> m_TriBaseIdx;

	// Number of nodes already in the BVH
	uint32_t m_UsedNodes = 0;
	uint32_t m_UsedIndices = 0;
};