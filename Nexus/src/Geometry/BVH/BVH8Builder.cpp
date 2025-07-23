#include "BVH8Builder.h"

#include <cstring>
#include <cassert>
#include "Utils/Utils.h"

BVH8Builder::BVH8Builder(const NXB::BVH2& bvh2) : m_Bvh2(bvh2)
{
}

void BVH8Builder::Init()
{
    m_Evals = std::vector<std::vector<NodeEval>>(m_Bvh2.nodeCount, std::vector<NodeEval>(7));
    m_TriCount = std::vector<int>(m_Bvh2.nodeCount);
    m_TriBaseIdx = std::vector<int>(m_Bvh2.nodeCount);
    ComputeNodeTriCount(0, 0);
    float rootCost = ComputeNodeCost(0, 0);
	//std::cout << rootCost << std::endl;
}

NXB::BVH8 BVH8Builder::Build()
{
    m_UsedNodes = 1;
    NXB::BVH8 bvh8;
    bvh8.nodes = new NXB::BVH8::Node[(4 * m_Bvh2.primCount - 1) / 7 + 1];
    CollapseNode(bvh8, 0, 0);
    //std::cout << "Used nodes: " << m_UsedNodes << std::endl;
    return bvh8;
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

float BVH8Builder::CInternal(const NXB::BVH2::Node& node, int& leftCount, int&rightCount)
{
    return CDistribute(node, 7, leftCount, rightCount) + node.bounds.Area() * C_NODE;
}

float BVH8Builder::ComputeNodeCost(uint32_t nodeIdx, int i)
{
    if (m_Evals[nodeIdx][i].decision != Decision::UNDEFINED)
        return m_Evals[nodeIdx][i].cost;

    const NXB::BVH2::Node& node = m_Bvh2.nodes[nodeIdx];

    if (node.leftChild == INVALID_IDX)
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

int BVH8Builder::ComputeNodeTriCount(int nodeIdx, int triBaseIdx)
{
    NXB::BVH2::Node& node = m_Bvh2.nodes[nodeIdx];

    if (node.leftChild == INVALID_IDX)
        m_TriCount[nodeIdx] = 1;
    else
    {
        int leftCount = ComputeNodeTriCount(node.leftChild, triBaseIdx);
        int rightCount = ComputeNodeTriCount(node.rightChild, leftCount);
        m_TriCount[nodeIdx] = leftCount + rightCount;
    }

    m_TriBaseIdx[nodeIdx] = triBaseIdx;

    return m_TriCount[nodeIdx];
}

void BVH8Builder::GetChildrenIndices(uint32_t nodeIdxBvh2, int* indices, int i, int& indicesCount)
{
	const NodeEval& eval = m_Evals[nodeIdxBvh2][i];

    // If in the first call the node is a leaf, return
	if (eval.decision == Decision::LEAF)
	{
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
		indices[indicesCount++] = node.leftChild;  // We reached a BVH8 internal node or leaf => stop recursion

	if (rightEval.decision == Decision::DISTRIBUTE)
		GetChildrenIndices(node.rightChild, indices, rightCount, indicesCount);
	else
		indices[indicesCount++] = node.rightChild;   // We reached a BVH8 internal node or leaf => stop recursion
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

int BVH8Builder::CountTriangles(NXB::BVH8& bvh8, uint32_t nodeIdxBvh2)
{
	const NXB::BVH2::Node& bvh2Node = m_Bvh2.nodes[nodeIdxBvh2];

	if (bvh2Node.leftChild == INVALID_IDX)
    {
        bvh8.primIdx[m_UsedIndices++] = bvh2Node.rightChild;
		return 1;
	}

	return CountTriangles(bvh8, bvh2Node.leftChild) + CountTriangles(bvh8, bvh2Node.rightChild);
}


void BVH8Builder::CollapseNode(NXB::BVH8& bvh8, uint32_t nodeIdxBvh2, uint32_t nodeIdxBvh8)
{
    const NXB::BVH2::Node& bvh2Node = m_Bvh2.nodes[nodeIdxBvh2];

    NXB::BVH8::NodeExplicit& bvh8Node = *(NXB::BVH8::NodeExplicit*)&bvh8.nodes[nodeIdxBvh8];

    const float denom = 1.0f / (float)((1 << N_Q) - 1);
    
    // e along each axis
    const float ex = ceilf(log2f((bvh2Node.bounds.bMax.x - bvh2Node.bounds.bMin.x) * denom));
    const float ey = ceilf(log2f((bvh2Node.bounds.bMax.y - bvh2Node.bounds.bMin.y) * denom));
    const float ez = ceilf(log2f((bvh2Node.bounds.bMax.z - bvh2Node.bounds.bMin.z) * denom));

    float exe = exp2f(ex);
    float eye = exp2f(ey);
    float eze = exp2f(ez);

    bvh8Node.e[0] = *(uint32_t*)&exe >> 23;
    bvh8Node.e[1] = *(uint32_t*)&eye >> 23;
    bvh8Node.e[2] = *(uint32_t*)&eze >> 23;

    bvh8Node.childBaseIdx = m_UsedNodes;
    bvh8Node.primBaseIdx = m_UsedIndices;

    bvh8Node.p = bvh2Node.bounds.bMin;
    bvh8Node.imask = 0;

    int childrenIndices[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    int indicesCount = 0;

    // Fill the array of children indices
	GetChildrenIndices(nodeIdxBvh2, childrenIndices, 0, indicesCount);

    // Order the children according to the octant traversal order
    OrderChildren(nodeIdxBvh2, childrenIndices);

    // Sum of triangles number in the node
    int nTrianglesTotal = 0;

	const float scaleX = 1.0f / powf(2, ex);
	const float scaleY = 1.0f / powf(2, ey);
	const float scaleZ = 1.0f / powf(2, ez);

    for (int i = 0; i < 8; i++)
    {
        if (childrenIndices[i] == -1)
        {
            // Empty child slot, set meta to 0
            bvh8Node.meta[i] = 0;
            continue;
        }
        else
        {
            const NXB::BVH2::Node& childNode = m_Bvh2.nodes[childrenIndices[i]];
			// Since the children are either internal or leaf nodes, we take their evaluation for i = 1
			const NodeEval& eval = m_Evals[childrenIndices[i]][0];
            assert(eval.decision != Decision::UNDEFINED);

            // Encode the child's bounding box origin
            bvh8Node.qlox[i] = static_cast<uint8_t>(floorf((childNode.bounds.bMin.x - bvh8Node.p.x) * scaleX));
            bvh8Node.qloy[i] = static_cast<uint8_t>(floorf((childNode.bounds.bMin.y - bvh8Node.p.y) * scaleY));
            bvh8Node.qloz[i] = static_cast<uint8_t>(floorf((childNode.bounds.bMin.z - bvh8Node.p.z) * scaleZ));

            // Encode the child's bounding box end point
            bvh8Node.qhix[i] = static_cast<uint8_t>(ceilf((childNode.bounds.bMax.x - bvh8Node.p.x) * scaleX));
            bvh8Node.qhiy[i] = static_cast<uint8_t>(ceilf((childNode.bounds.bMax.y - bvh8Node.p.y) * scaleY));
            bvh8Node.qhiz[i] = static_cast<uint8_t>(ceilf((childNode.bounds.bMax.z - bvh8Node.p.z) * scaleZ));

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
                const int nTriangles = CountTriangles(bvh8, childrenIndices[i]);
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
    }
	assert(nTrianglesTotal <= 24);


    // Caching child base index before resizing nodes array
    uint32_t childBaseIdx = bvh8Node.childBaseIdx;

    //bvh8.nodes.resize(m_UsedNodes);

    int childCount = 0;
    // Recursively collapse internal children nodes
    for (int i = 0; i < 8; i++)
    {
        if (childrenIndices[i] == -1)
            continue;

        const NodeEval& eval = m_Evals[childrenIndices[i]][0];

        if (eval.decision == Decision::INTERNAL)
        {
            CollapseNode(bvh8, childrenIndices[i], childBaseIdx + childCount);
            childCount++;
        }
    }
}

