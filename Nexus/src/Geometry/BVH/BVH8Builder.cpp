#include "BVH8Builder.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include "Utils/Utils.h"

namespace {

	/* \brief Biased exponent of the smallest power of two >= x
	 *
	 * Mirrors NXB::CeilLog2 in WideConverter.cu bit for bit: the quantization grid has to
	 * agree with the device builder's for the two outputs to be comparable. It also
	 * behaves where the obvious ceilf(log2f(x)) does not. A mesh flat on one axis has a
	 * zero diagonal there, and this returns 0, whose inverse scale below is a finite
	 * 2^127 rather than the infinity that would quantize every bound on that axis to NaN.
	 */
	uint8_t CeilLog2(float x)
	{
		uint32_t ix;
		std::memcpy(&ix, &x, sizeof(ix));

		const uint32_t exponent = (ix >> 23) & 0xff;

		// x is exactly a power of two if its mantissa bits are zero
		const bool isPow2 = (ix & ((1 << 23) - 1)) == 0;

		return static_cast<uint8_t>(exponent + (isPow2 ? 0 : 1));
	}

	// 2^-e for the biased exponent e produced by CeilLog2
	float InvPow2(uint8_t eBiased)
	{
		const uint32_t bits = static_cast<uint32_t>(254 - eBiased) << 23;

		float result;
		std::memcpy(&result, &bits, sizeof(result));
		return result;
	}
}

namespace Nexus {

	BVH8Builder::BVH8Builder(NXB::BVH2::Host bvh2) : m_Bvh2(std::move(bvh2))
	{
	}

	void BVH8Builder::Init()
	{
		const uint32_t nodeCount = static_cast<uint32_t>(m_Bvh2.nodes.size());

		if (nodeCount == 0)
			return;

		m_Evals = std::vector<std::vector<NodeEval>>(nodeCount, std::vector<NodeEval>(7));
		m_TriCount = std::vector<int>(nodeCount);

		// NXB merges bottom up, so the root is the last node written rather than node 0.
		// BuildBVH8 seeds its own collapse with nodeCount - 1 for the same reason.
		m_RootIdx = nodeCount - 1;

		ComputeNodeTriCount(m_RootIdx);
		float rootCost = ComputeNodeCost(m_RootIdx, 0);
		//std::cout << rootCost << std::endl;
	}

	NXB::BVH8 BVH8Builder::Build(cudaStream_t stream)
	{
		if (m_Bvh2.nodes.empty())
			return NXB::BVH8();

		if (m_Evals.empty())
			Init();

		/*
		 * Worst case node count of the collapse, the same (4n - 1) / 7 bound BuildBVH8
		 * allocates: it is reached when every internal node above the leaves holds just
		 * two of them. Packing up to P_MAX primitives per leaf can only merge further, so
		 * the bound still holds here.
		 */
		const size_t allocatedNodeCount = (4ull * m_Bvh2.primCount + 5) / 7;

		m_Nodes.assign(allocatedNodeCount, NXB::BVH8::Node{});
		m_PrimIdx.assign(m_Bvh2.primCount, 0);
		m_UsedNodes = 1;
		m_UsedIndices = 0;

		CollapseNode(m_RootIdx, 0);

		assert(m_UsedNodes <= allocatedNodeCount);
		assert(m_UsedIndices == m_Bvh2.primCount);

		/*
		 * The array is allocated at the worst case and only the nodes the collapse
		 * produced are uploaded, so NodeCount() and the allocation differ exactly as they
		 * do for a device build, which is what BVH8 expects.
		 */
		NXB::DeviceBuffer<NXB::BVH8::Node> nodes(allocatedNodeCount, stream);
		nodes.Upload(m_Nodes.data(), m_UsedNodes);

		NXB::DeviceBuffer<uint32_t> primIdx(m_PrimIdx, stream);

		return NXB::BVH8(std::move(nodes), std::move(primIdx), m_UsedNodes, m_Bvh2.primCount, m_Bvh2.bounds);
	}

	float BVH8Builder::CLeaf(const NXB::BVH2::Node& node, int triCount)
	{
		if (triCount > P_MAX)
			return 1.0e30f;

		return node.bounds.Area() * triCount * C_PRIM;
	}

	float BVH8Builder::CDistribute(const NXB::BVH2::Node& node, int j, int& leftCount, int& rightCount)
	{
		float cDistribute = 1.0e30f;

		// k in (1 .. j - 1) in the paper
		for (int k = 0; k < j; k++)
		{
			const float cLeft = ComputeNodeCost(node.leftChild, k);
			const float cRight = ComputeNodeCost(node.rightChild, j - 1 - k);

			if (cLeft + cRight < cDistribute)
			{
				cDistribute = cLeft + cRight;
				leftCount = k;
				rightCount = j - 1 - k;
			}
		}
		return cDistribute;
	}

	float BVH8Builder::CInternal(const NXB::BVH2::Node& node, int& leftCount, int& rightCount)
	{
		return CDistribute(node, 7, leftCount, rightCount) + node.bounds.Area() * C_NODE;
	}

	float BVH8Builder::ComputeNodeCost(uint32_t nodeIdx, int i)
	{
		if (m_Evals[nodeIdx][i].decision != Decision::UNDEFINED)
			return m_Evals[nodeIdx][i].cost;

		const NXB::BVH2::Node& node = m_Bvh2.nodes[nodeIdx];

		if (node.leftChild == NXB::InvalidIdx)
		{
			// TODO: can be optimized by setting all costs for i in (0 .. 6) to cLeaf
			m_Evals[nodeIdx][i].decision = Decision::LEAF;
			m_Evals[nodeIdx][i].cost = CLeaf(node, 1);

			return m_Evals[nodeIdx][i].cost;
		}

		// i = 1 in the paper
		if (i == 0)
		{
			int leftCount, rightCount;
			const float cLeaf = CLeaf(node, m_TriCount[nodeIdx]);
			const float cInternal = CInternal(node, leftCount, rightCount);

			if (cLeaf < cInternal)
			{
				m_Evals[nodeIdx][i].decision = Decision::LEAF;
				m_Evals[nodeIdx][i].cost = cLeaf;
			}
			else
			{
				m_Evals[nodeIdx][i].decision = Decision::INTERNAL;
				m_Evals[nodeIdx][i].cost = cInternal;
				m_Evals[nodeIdx][i].leftCount = leftCount;
				m_Evals[nodeIdx][i].rightCount = rightCount;
			}
			return m_Evals[nodeIdx][i].cost;
		}

		// i in (2 .. 7) in the paper
		int leftCount, rightCount;
		const float cDistribute = CDistribute(node, i, leftCount, rightCount);
		const float cFewerRoots = ComputeNodeCost(nodeIdx, i - 1);

		if (cDistribute < cFewerRoots)
		{
			m_Evals[nodeIdx][i].decision = Decision::DISTRIBUTE;
			m_Evals[nodeIdx][i].cost = cDistribute;
			m_Evals[nodeIdx][i].leftCount = leftCount;
			m_Evals[nodeIdx][i].rightCount = rightCount;
		}
		else
			m_Evals[nodeIdx][i] = m_Evals[nodeIdx][i - 1];

		return m_Evals[nodeIdx][i].cost;
	}

	int BVH8Builder::ComputeNodeTriCount(uint32_t nodeIdx)
	{
		const NXB::BVH2::Node& node = m_Bvh2.nodes[nodeIdx];

		if (node.leftChild == NXB::InvalidIdx)
			m_TriCount[nodeIdx] = 1;
		else
			m_TriCount[nodeIdx] = ComputeNodeTriCount(node.leftChild) + ComputeNodeTriCount(node.rightChild);

		return m_TriCount[nodeIdx];
	}

	void BVH8Builder::GetChildrenIndices(uint32_t nodeIdxBvh2, int* indices, int i, int& indicesCount)
	{
		const NodeEval& eval = m_Evals[nodeIdxBvh2][i];

		// If in the first call the node is a leaf, return
		if (eval.decision == Decision::LEAF)
		{
			assert(indicesCount < 8);
			indices[indicesCount++] = nodeIdxBvh2;
			return;
		}

		// Decision is either INTERNAL or DISTRIBUTE
		const NXB::BVH2::Node& node = m_Bvh2.nodes[nodeIdxBvh2];

		const int leftCount = eval.leftCount;
		const int rightCount = eval.rightCount;

		// Retreive the decision for the left and right childs
		const NodeEval& leftEval = m_Evals[node.leftChild][leftCount];
		const NodeEval& rightEval = m_Evals[node.rightChild][rightCount];

		// Recurse in child nodes if we need to distribute
		if (leftEval.decision == Decision::DISTRIBUTE)
			GetChildrenIndices(node.leftChild, indices, leftCount, indicesCount);
		else
		{
			// We reached a BVH8 internal node or leaf => stop recursion
			assert(indicesCount < 8);
			indices[indicesCount++] = node.leftChild;
		}

		if (rightEval.decision == Decision::DISTRIBUTE)
			GetChildrenIndices(node.rightChild, indices, rightCount, indicesCount);
		else
		{
			// We reached a BVH8 internal node or leaf => stop recursion
			assert(indicesCount < 8);
			indices[indicesCount++] = node.rightChild;
		}
	}

	void BVH8Builder::OrderChildren(uint32_t nodeIdxBvh2, int* childrenIndices)
	{
		const NXB::BVH2::Node& parentNode = m_Bvh2.nodes[nodeIdxBvh2];
		const float3 parentCentroid = (parentNode.bounds.bMax + parentNode.bounds.bMin) * 0.5f;

		// Fill the table cost(c, s)
		float cost[8][8];
		int childCount = 0;

		for (int c = 0; c < 8; c++)
		{
			// If no more children, break
			if (childrenIndices[c] == -1)
				break;

			for (int s = 0; s < 8; s++)
			{
				// Ray direction
				const float dsx = (s & 0b100) ? -1.0f : 1.0f;
				const float dsy = (s & 0b010) ? -1.0f : 1.0f;
				const float dsz = (s & 0b001) ? -1.0f : 1.0f;
				const float3 ds = make_float3(dsx, dsy, dsz);

				const NXB::BVH2::Node& childNode = m_Bvh2.nodes[childrenIndices[c]];
				const float3 centroid = (childNode.bounds.bMin + childNode.bounds.bMax) * 0.5f;
				cost[c][s] = dot(centroid - parentCentroid, ds);
			}
			childCount++;
		}

		// Greedy ordering
		// TODO: implement auction algorithm?
		// See https://dspace.mit.edu/bitstream/handle/1721.1/3233/P-2064-24690022.pdf

		bool slotAssigned[8] = { 0 };
		int assignment[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };

		while (true)
		{
			float minCost = std::numeric_limits<float>::max();
			int assignedNode = -1, assignedSlot = -1;

			for (int c = 0; c < childCount; c++)
			{
				// If node already assigned, skip
				if (assignment[c] != -1)
					continue;

				for (int s = 0; s < 8; s++)
				{
					// If slot already used, skip
					if (slotAssigned[s])
						continue;

					if (cost[c][s] < minCost)
					{
						minCost = cost[c][s];
						assignedNode = c;
						assignedSlot = s;
					}
				}
			}

			// If all the nodes have been assigned
			if (assignedNode == -1)
				break;

			// Assign the node to the specific position
			assignment[assignedNode] = assignedSlot;
			slotAssigned[assignedSlot] = true;
		}

		int indicesCpy[8];
		memcpy(indicesCpy, childrenIndices, 8 * sizeof(int));

		for (int i = 0; i < 8; i++)
			childrenIndices[i] = -1;

		// Reorder the nodes
		for (int i = 0; i < childCount; i++)
			childrenIndices[assignment[i]] = indicesCpy[i];

	}

	int BVH8Builder::CountTriangles(uint32_t nodeIdxBvh2)
	{
		const NXB::BVH2::Node& bvh2Node = m_Bvh2.nodes[nodeIdxBvh2];

		if (bvh2Node.leftChild == NXB::InvalidIdx)
		{
			// rightChild holds the primitive index on a leaf
			assert(m_UsedIndices < m_PrimIdx.size());
			m_PrimIdx[m_UsedIndices++] = bvh2Node.rightChild;
			return 1;
		}

		return CountTriangles(bvh2Node.leftChild) + CountTriangles(bvh2Node.rightChild);
	}


	void BVH8Builder::CollapseNode(uint32_t nodeIdxBvh2, uint32_t nodeIdxBvh8)
	{
		assert(nodeIdxBvh8 < m_Nodes.size());

		const NXB::BVH2::Node& bvh2Node = m_Bvh2.nodes[nodeIdxBvh2];

		// The readable view of the packed node, written back into m_Nodes below
		NXB::BVH8::NodeExplicit bvh8Node = { };

		const float quantStep = 1.0f / (float)((1 << N_Q) - 1);
		const float3 diagonal = bvh2Node.bounds.bMax - bvh2Node.bounds.bMin;

		bvh8Node.p = bvh2Node.bounds.bMin;
		bvh8Node.imask = 0;

		// e along each axis
		bvh8Node.e[0] = CeilLog2(diagonal.x * quantStep);
		bvh8Node.e[1] = CeilLog2(diagonal.y * quantStep);
		bvh8Node.e[2] = CeilLog2(diagonal.z * quantStep);

		bvh8Node.childBaseIdx = m_UsedNodes;
		bvh8Node.primBaseIdx = m_UsedIndices;

		const float3 invE = make_float3(InvPow2(bvh8Node.e[0]), InvPow2(bvh8Node.e[1]), InvPow2(bvh8Node.e[2]));

		int childrenIndices[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
		int indicesCount = 0;

		// Fill the array of children indices
		GetChildrenIndices(nodeIdxBvh2, childrenIndices, 0, indicesCount);

		// Order the children according to the octant traversal order
		OrderChildren(nodeIdxBvh2, childrenIndices);

		// Sum of triangles number in the node
		int nTrianglesTotal = 0;

		for (int i = 0; i < 8; i++)
		{
			if (childrenIndices[i] == -1)
			{
				/*
				 * Empty child slot. meta[i] == 0 is already enough for traversal to skip
				 * it, but leaving the quantized bounds unwritten would upload whatever
				 * the allocation happened to hold, so store an empty box the way
				 * BuildBVH8 does.
				 */
				bvh8Node.meta[i] = 0;
				bvh8Node.qlox[i] = bvh8Node.qloy[i] = bvh8Node.qloz[i] = 0xff;
				bvh8Node.qhix[i] = bvh8Node.qhiy[i] = bvh8Node.qhiz[i] = 0x00;
				continue;
			}

			const NXB::BVH2::Node& childNode = m_Bvh2.nodes[childrenIndices[i]];
			// Since the children are either internal or leaf nodes, we take their evaluation for i = 1
			const NodeEval& eval = m_Evals[childrenIndices[i]][0];
			assert(eval.decision != Decision::UNDEFINED);

			// Encode the child's bounding box origin
			bvh8Node.qlox[i] = static_cast<uint8_t>(floorf((childNode.bounds.bMin.x - bvh8Node.p.x) * invE.x));
			bvh8Node.qloy[i] = static_cast<uint8_t>(floorf((childNode.bounds.bMin.y - bvh8Node.p.y) * invE.y));
			bvh8Node.qloz[i] = static_cast<uint8_t>(floorf((childNode.bounds.bMin.z - bvh8Node.p.z) * invE.z));

			// Encode the child's bounding box end point
			bvh8Node.qhix[i] = static_cast<uint8_t>(ceilf((childNode.bounds.bMax.x - bvh8Node.p.x) * invE.x));
			bvh8Node.qhiy[i] = static_cast<uint8_t>(ceilf((childNode.bounds.bMax.y - bvh8Node.p.y) * invE.y));
			bvh8Node.qhiz[i] = static_cast<uint8_t>(ceilf((childNode.bounds.bMax.z - bvh8Node.p.z) * invE.z));

			if (eval.decision == Decision::INTERNAL)
			{
				m_UsedNodes++;
				// High 3 bits to 001
				bvh8Node.meta[i] = 0b00100000;
				// Low 5 bits to 24 + child index
				bvh8Node.meta[i] |= 24 + i;
				// Set the child node as an internal node in the imask field
				bvh8Node.imask |= 1 << i;
			}
			else if (eval.decision == Decision::LEAF)
			{
				const int nTriangles = CountTriangles(childrenIndices[i]);
				assert(nTriangles <= P_MAX);

				bvh8Node.meta[i] = 0;

				// High 3 bits store the number of triangles in unary encoding
				for (int j = 0; j < nTriangles; j++)
				{
					bvh8Node.meta[i] |= 1 << (j + 5);
				}
				// Low 5 bits store the index of first triangle relative to the triangle base index
				bvh8Node.meta[i] |= nTrianglesTotal;

				nTrianglesTotal += nTriangles;
			}
		}
		assert(nTrianglesTotal <= 24);

		static_assert(sizeof(NXB::BVH8::NodeExplicit) == sizeof(NXB::BVH8::Node),
			"NodeExplicit is the readable view of the packed node and has to match its size");
		std::memcpy(&m_Nodes[nodeIdxBvh8], &bvh8Node, sizeof(bvh8Node));

		// Caching child base index, the loop above bumped m_UsedNodes for every internal child
		const uint32_t childBaseIdx = bvh8Node.childBaseIdx;

		int childCount = 0;
		// Recursively collapse internal children nodes
		for (int i = 0; i < 8; i++)
		{
			if (childrenIndices[i] == -1)
				continue;

			const NodeEval& eval = m_Evals[childrenIndices[i]][0];

			if (eval.decision == Decision::INTERNAL)
			{
				CollapseNode(childrenIndices[i], childBaseIdx + childCount);
				childCount++;
			}
		}
	}

}
