#pragma once
#include "BVH8.h"

#include <iostream>

#include "BVH2.h"
#include "TLAS.h"

/*
 * Class used to convert a TLAS from a BVH2 to a BVH8 layout
 */
class TLASBuilder
{
public:

	TLASBuilder(TLAS& tlas);

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

	BVH8 Build();
	float ComputeNodeCost(uint32_t nodeIdx, int i);
	void Init();

	void CollapseNode(BVH8& bvh8, uint32_t nodeIdxBvh2, uint32_t nodeIdxBvh8);

private:

	// Cleaf(n)
	inline float CLeaf(const TLASNode& node, int triCount);

	// Cinternal(n)
	float CInternal(const TLASNode& node, int& leftCount, int& rightCount);

	// Cdistribute(n, j)
	float CDistribute(const TLASNode& node, int j, int& leftCount, int& rightCount);

	// Returns the indices of the node's children
	void GetChildrenIndices(uint32_t nodeIdxBvh2, int *indices, int i, int& indicesCount);

	int CountTriangles(BVH8& bvh8, uint32_t nodeIdxBvh2);

	// Order the children in a given node
	void OrderChildren(uint32_t nodeIdxBvh2, int* childrenIndices);

private:
	TLAS& m_Tlas;

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
