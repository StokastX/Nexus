#pragma once

#include <cstdint>
#include <vector>
#include "BVH.h"

#define C_PRIM 0.3f  // Cost of a ray-primitive intersection
#define C_NODE 1.0f  // Cost of a ray-node intersection
#define P_MAX  3	 // Maximum allowed leaf size
#define N_Q 8		 // Number of bits used to store the childs' AABB coordinates

namespace Nexus {

	/* \brief Reference CPU implementation of the BVH8 collapse (Ylitie et al.)
	 *
	 * NXB::BuildBVH8 does this on the device, and does it orders of magnitude faster. What
	 * this is for is the single threaded ground truth: a readable tree to diff against the
	 * device output when a traversal bug shows up, and a place to prototype collapse
	 * heuristics (see the auction TODO in OrderChildren) before writing them as a kernel.
	 * Nothing on the render path calls it.
	 *
	 *     NXB::BVH2 bvh2 = NXB::BuildBVH2<NXB::Triangle>(triangles, triangleCount);
	 *     BVH8Builder builder(bvh2.ToHost());
	 *     NXB::BVH8 bvh8 = builder.Build();
	 *
	 * The encoding matches BuildBVH8's bit for bit, so BVH8Traversal.cuh traverses either
	 * without knowing which produced it and NXB::ComputeSAHCost compares the two directly.
	 * The one deliberate difference is leaf size: the device collapse puts a single
	 * primitive in every leaf slot, this one packs up to P_MAX. That also makes
	 * NXB::BVH8::AverageChildPerNode meaningless for this builder's output, since its
	 * formula assumes one primitive per leaf.
	 *
	 * Cost, and why this is debugging-only: m_Evals is nodeCount x 7 entries as a vector
	 * of vectors, and both the cost evaluation and the collapse recurse once per node. A
	 * million triangle mesh is ~2M nodes, i.e. hundreds of MB of evaluations and a
	 * recursion deep enough to overflow the stack on an unbalanced tree.
	 */
	class BVH8Builder
	{
	public:

		/* \param bvh2 Host copy of the binary BVH to collapse, from NXB::BVH2::ToHost().
		 *        Taken by value so passing that call's result directly moves rather than
		 *        leaving a reference dangling on a temporary.
		 */
		BVH8Builder(NXB::BVH2::Host bvh2);

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

		/* \brief Evaluates the SAH cost and the collapse decision of every BVH2 node
		 *
		 * Build calls this itself. It is public so a caller can run it alone and read
		 * GetEvals() to inspect the decisions without paying for the collapse.
		 */
		void Init();

		/* \brief Collapses the binary BVH into a wide one
		 *
		 * \param stream The stream the upload is issued on, and the one the returned BVH8
		 *        releases its memory on
		 *
		 * \returns The wide BVH, owning its device memory, empty if the input was empty
		 */
		NXB::BVH8 Build(cudaStream_t stream = 0);

		const std::vector<std::vector<NodeEval>>& GetEvals() const { return m_Evals; }

		// Index of the BVH2 root. NXB merges bottom up, so it is the last node, not node 0
		uint32_t GetRootIdx() const { return m_RootIdx; }

	private:

		// Cleaf(n)
		inline float CLeaf(const NXB::BVH2::Node& node, int triCount);

		// Cinternal(n)
		float CInternal(const NXB::BVH2::Node& node, int& leftCount, int& rightCount);

		// Cdistribute(n, j)
		float CDistribute(const NXB::BVH2::Node& node, int j, int& leftCount, int& rightCount);

		// Number of triangles in the subtree of nodeIdx
		int ComputeNodeTriCount(uint32_t nodeIdx);

		float ComputeNodeCost(uint32_t nodeIdx, int i);

		// Returns the indices of the node's children
		void GetChildrenIndices(uint32_t nodeIdxBvh2, int* indices, int i, int& indicesCount);

		// Appends the subtree's primitive indices to m_PrimIdx and returns how many
		int CountTriangles(uint32_t nodeIdxBvh2);

		// Order the children in a given node
		void OrderChildren(uint32_t nodeIdxBvh2, int* childrenIndices);

		void CollapseNode(uint32_t nodeIdxBvh2, uint32_t nodeIdxBvh8);

	private:
		NXB::BVH2::Host m_Bvh2;

		// Index of the BVH2 root node
		uint32_t m_RootIdx = 0;

		// Optimal SAH cost C(n, i) with decisions
		std::vector<std::vector<NodeEval>> m_Evals;

		// Number of triangles in the subtree of the node i
		std::vector<int> m_TriCount;

		// The wide BVH being assembled on the host, uploaded by Build
		std::vector<NXB::BVH8::Node> m_Nodes;
		std::vector<uint32_t> m_PrimIdx;

		// Number of nodes already in the BVH
		uint32_t m_UsedNodes = 0;
		uint32_t m_UsedIndices = 0;
	};

}
