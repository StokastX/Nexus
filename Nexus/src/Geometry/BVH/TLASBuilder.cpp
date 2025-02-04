#include "TLASBuilder.h"

//TLASBuilder::TLASBuilder(TLAS& tlas) : m_Tlas(tlas)
//{
//    m_Evals = std::vector<std::vector<NodeEval>>(m_Tlas.nodes.size(), std::vector<NodeEval>(7));
//    m_TriCount = std::vector<int>(m_Tlas.nodes.size());
//    m_TriBaseIdx = std::vector<int>(m_Tlas.nodes.size());
//}
//
//void TLASBuilder::Init()
//{
//    float rootCost = ComputeNodeCost(0, 0);
//	//std::cout << rootCost << std::endl;
//}
//
//BVH8 TLASBuilder::Build()
//{
//    m_UsedNodes = 1;
//    BVH8 bvh8;
//    bvh8.triangleIdx = m_Tlas.instancesIdx;
//    bvh8.nodes.emplace_back();
//    CollapseNode(bvh8, 0, 0);
//    //std::cout << "Used nodes: " << m_UsedNodes << std::endl;
//    return bvh8;
//}
//
//float TLASBuilder::CLeaf(const TLASNode& node, int blasCount)
//{
//    if (blasCount > P_MAX)
//		return 1.0e30f;
//
//    AABB nodeAABB(node.aabbMin, node.aabbMax);
//    return nodeAABB.Area() * blasCount * C_PRIM;
//}
//
//float TLASBuilder::CDistribute(const TLASNode& node, int j, int& leftCount, int& rightCount)
//{
//    float cDistribute = 1.0e30f;
//
//    // k in (1 .. j - 1) in the paper
//	for (int k = 0; k < j; k++)
//	{
//        const float cLeft = ComputeNodeCost(node.left, k);
//        const float cRight = ComputeNodeCost(node.right, j - 1 - k);
//
//        if (cLeft + cRight < cDistribute)
//        {
//            cDistribute = cLeft + cRight;
//            leftCount = k;
//            rightCount = j - 1 - k;
//        }
//	}
//    return cDistribute;
//}
//
//float TLASBuilder::CInternal(const TLASNode& node, int& leftCount, int&rightCount)
//{
//    AABB nodeAABB(node.aabbMin, node.aabbMax);
//    return CDistribute(node, 7, leftCount, rightCount) + nodeAABB.Area() * C_NODE;
//}
//
//float TLASBuilder::ComputeNodeCost(uint32_t nodeIdx, int i)
//{
//    if (m_Evals[nodeIdx][i].decision != Decision::UNDEFINED)
//        return m_Evals[nodeIdx][i].cost;
//
//    const TLASNode& node = m_Tlas.nodes[nodeIdx];
//
//    if (node.IsLeaf())
//    {
//        // TODO: can be optimized by setting all costs for i in (0 .. 6) to cLeaf
//        m_Evals[nodeIdx][i].decision = Decision::LEAF;
//        m_Evals[nodeIdx][i].cost = CLeaf(node, node.blasCount);
//
//        return m_Evals[nodeIdx][i].cost;
//    }
//
//    // i = 1 in the paper
//    if (i == 0)
//    {
//        int leftCount, rightCount;
//        const float cLeaf = CLeaf(node, node.blasCount);
//        const float cInternal = CInternal(node, leftCount, rightCount);
//
//        if (cLeaf < cInternal)
//        {
//            m_Evals[nodeIdx][i].decision = Decision::LEAF;
//            m_Evals[nodeIdx][i].cost = cLeaf;
//        }
//        else
//        {
//            m_Evals[nodeIdx][i].decision = Decision::INTERNAL;
//            m_Evals[nodeIdx][i].cost = cInternal;
//			m_Evals[nodeIdx][i].leftCount = leftCount;
//			m_Evals[nodeIdx][i].rightCount = rightCount;
//        }
//        return m_Evals[nodeIdx][i].cost;
//    }
//
//    // i in (2 .. 7) in the paper
//    int leftCount, rightCount;
//    const float cDistribute = CDistribute(node, i, leftCount, rightCount);
//    const float cFewerRoots = ComputeNodeCost(nodeIdx, i - 1);
//
//    if (cDistribute < cFewerRoots)
//    {
//        m_Evals[nodeIdx][i].decision = Decision::DISTRIBUTE;
//        m_Evals[nodeIdx][i].cost = cDistribute;
//        m_Evals[nodeIdx][i].leftCount = leftCount;
//        m_Evals[nodeIdx][i].rightCount = rightCount;
//    }
//    else
//        m_Evals[nodeIdx][i] = m_Evals[nodeIdx][i - 1];
//
//    return m_Evals[nodeIdx][i].cost;
//}
//
//void TLASBuilder::GetChildrenIndices(uint32_t nodeIdxBvh2, int* indices, int i, int& indicesCount)
//{
//	const NodeEval& eval = m_Evals[nodeIdxBvh2][i];
//
//    // If in the first call the node is a leaf, return
//	if (eval.decision == Decision::LEAF)
//	{
//		indices[indicesCount++] = nodeIdxBvh2;
//		return;
//	}
//
//	// Decision is either INTERNAL or DISTRIBUTE
//	const TLASNode& node = m_Tlas.nodes[nodeIdxBvh2];
//
//	const int leftCount = eval.leftCount;
//	const int rightCount = eval.rightCount;
//
//	// Retreive the decision for the left and right childs
//	const NodeEval& leftEval = m_Evals[node.left][leftCount];
//	const NodeEval& rightEval = m_Evals[node.right][rightCount];
//
//	// Recurse in child nodes if we need to distribute
//	if (leftEval.decision == Decision::DISTRIBUTE)
//		GetChildrenIndices(node.left, indices, leftCount, indicesCount);
//	else
//		indices[indicesCount++] = node.left;   // We reached a BVH8 internal node or leaf => stop recursion
//
//	if (rightEval.decision == Decision::DISTRIBUTE)
//		GetChildrenIndices(node.right, indices, rightCount, indicesCount);
//	else
//		indices[indicesCount++] = node.right;   // We reached a BVH8 internal node or leaf => stop recursion
//}
//
//void TLASBuilder::OrderChildren(uint32_t nodeIdxBvh2, int* childrenIndices)
//{
//    const TLASNode& parentNode = m_Tlas.nodes[nodeIdxBvh2];
//    const float3 parentCentroid = (parentNode.aabbMax + parentNode.aabbMin) * 0.5f;
//
//    // Fill the table cost(c, s)
//    float cost[8][8];
//    int childCount = 0;
//
//    for (int c = 0; c < 8; c++)
//    {
//        // If no more children, break
//        if (childrenIndices[c] == -1)
//            break;
//
//        for (int s = 0; s < 8; s++)
//        {
//            // Ray direction
//            const float dsx = (s & 0b100) ? -1.0f : 1.0f;
//            const float dsy = (s & 0b010) ? -1.0f : 1.0f;
//            const float dsz = (s & 0b001) ? -1.0f : 1.0f;
//            const float3 ds = make_float3(dsx, dsy, dsz);
//
//            const TLASNode& childNode = m_Tlas.nodes[childrenIndices[c]];
//            const float3 centroid = (childNode.aabbMin + childNode.aabbMax) * 0.5f;
//            cost[c][s] = dot(centroid - parentCentroid, ds);
//        }
//        childCount++;
//    }
//
//    // Greedy ordering
//    // TODO: implement auction algorithm?
//    // See https://dspace.mit.edu/bitstream/handle/1721.1/3233/P-2064-24690022.pdf
//
//    bool slotAssigned[8] = { 0 };
//    int assignment[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
//
//    while (true)
//    {
//		float minCost = std::numeric_limits<float>::max();
//        int assignedNode = -1, assignedSlot = -1;
//
//        for (int c = 0; c < childCount; c++)
//        {
//            // If node already assigned, skip
//            if (assignment[c] != -1)
//                continue;
//
//            for (int s = 0; s < 8; s++)
//            {
//                // If slot already used, skip
//                if (slotAssigned[s])
//                    continue;
//
//                if (cost[c][s] < minCost)
//                {
//                    minCost = cost[c][s];
//                    assignedNode = c;
//                    assignedSlot = s;
//                }
//            }
//        }
//
//        // If all the nodes have been assigned
//        if (assignedNode == -1)
//            break;
//
//        // Assign the node to the specific position
//        assignment[assignedNode] = assignedSlot;
//        slotAssigned[assignedSlot] = true;
//    }
//
//    int indicesCpy[8];
//    memcpy(indicesCpy, childrenIndices, 8 * sizeof(int));
//
//    for (int i = 0; i < 8; i++)
//        childrenIndices[i] = -1;
//
//    // Reorder the nodes
//    for (int i = 0; i < childCount; i++)
//        childrenIndices[assignment[i]] = indicesCpy[i];
//
//}
//
//int TLASBuilder::CountTriangles(BVH8& bvh8, uint32_t nodeIdxTlas)
//{
//	const TLASNode& tlasNode = m_Tlas.nodes[nodeIdxTlas];
//
//	if (tlasNode.IsLeaf())
//    {
//		bvh8.triangleIdx[m_UsedIndices++] = tlasNode.blasIdx;
//		return 1;
//	}
//
//	return CountTriangles(bvh8, tlasNode.left) + CountTriangles(bvh8, tlasNode.right);
//}
//
//
//void TLASBuilder::CollapseNode(BVH8& bvh8, uint32_t nodeIdxTlas, uint32_t nodeIdxBvh8)
//{
//    const TLASNode& tlasNode = m_Tlas.nodes[nodeIdxTlas];
//
//    BVH8Node& bvh8Node = bvh8.nodes[nodeIdxBvh8];
//
//    const float denom = 1.0f / (float)((1 << N_Q) - 1);
//    
//    // e along each axis
//    const float ex = ceilf(log2f((tlasNode.aabbMax.x - tlasNode.aabbMin.x) * denom));
//    const float ey = ceilf(log2f((tlasNode.aabbMax.y - tlasNode.aabbMin.y) * denom));
//    const float ez = ceilf(log2f((tlasNode.aabbMax.z - tlasNode.aabbMin.z) * denom));
//
//    float exe = exp2f(ex);
//    float eye = exp2f(ey);
//    float eze = exp2f(ez);
//
//    bvh8Node.e[0] = *(uint32_t*)&exe >> 23;
//    bvh8Node.e[1] = *(uint32_t*)&eye >> 23;
//    bvh8Node.e[2] = *(uint32_t*)&eze >> 23;
//
//    bvh8Node.childBaseIdx = m_UsedNodes;
//    bvh8Node.triangleBaseIdx = m_UsedIndices;
//
//    bvh8Node.p = tlasNode.aabbMin;
//    bvh8Node.imask = 0;
//
//    int childrenIndices[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
//    int indicesCount = 0;
//
//    // Fill the array of children indices
//	GetChildrenIndices(nodeIdxTlas, childrenIndices, 0, indicesCount);
//
//    // Order the children according to the octant traversal order
//    OrderChildren(nodeIdxTlas, childrenIndices);
//
//    // Sum of triangles number in the node
//    int nTrianglesTotal = 0;
//
//	const float scaleX = 1.0f / powf(2, ex);
//	const float scaleY = 1.0f / powf(2, ey);
//	const float scaleZ = 1.0f / powf(2, ez);
//
//    for (int i = 0; i < 8; i++)
//    {
//        if (childrenIndices[i] == -1)
//        {
//            // Empty child slot, set meta to 0
//            bvh8Node.meta[i] = 0;
//            continue;
//        }
//        else
//        {
//            const TLASNode& childNode = m_Tlas.nodes[childrenIndices[i]];
//			// Since the children are either internal or leaf nodes, we take their evaluation for i = 1
//			const NodeEval& eval = m_Evals[childrenIndices[i]][0];
//            assert(eval.decision != Decision::UNDEFINED);
//
//            // Encode the child's bounding box origin
//            bvh8Node.qlox[i] = static_cast<byte>(floorf((childNode.aabbMin.x - bvh8Node.p.x) * scaleX));
//            bvh8Node.qloy[i] = static_cast<byte>(floorf((childNode.aabbMin.y - bvh8Node.p.y) * scaleY));
//            bvh8Node.qloz[i] = static_cast<byte>(floorf((childNode.aabbMin.z - bvh8Node.p.z) * scaleZ));
//
//            // Encode the child's bounding box end point
//            const float qhix = ceilf((childNode.aabbMax.x - bvh8Node.p.x) * scaleX);
//            bvh8Node.qhix[i] = static_cast<byte>(std::min(ceilf((childNode.aabbMax.x - bvh8Node.p.x) * scaleX), 255.0f));
//            bvh8Node.qhiy[i] = static_cast<byte>(std::min(ceilf((childNode.aabbMax.y - bvh8Node.p.y) * scaleY), 255.0f));
//            bvh8Node.qhiz[i] = static_cast<byte>(std::min(ceilf((childNode.aabbMax.z - bvh8Node.p.z) * scaleZ), 255.0f));
//
//            if (eval.decision == Decision::INTERNAL)
//            {
//				m_UsedNodes++;
//                // High 3 bits to 001
//                bvh8Node.meta[i] = 0b00100000;
//                // Low 5 bits to 24 + child index
//                bvh8Node.meta[i] |= 24 + i;
//                // Set the child node as an internal node in the imask field
//                bvh8Node.imask |= 1 << i;
//            }
//            else if (eval.decision == Decision::LEAF)
//            {
//                const int nTriangles =  CountTriangles(bvh8, childrenIndices[i]);
//                assert(nTriangles <= P_MAX);
//
//                bvh8Node.meta[i] = 0;
//
//                // High 3 bits store the number of triangles in unary encoding
//                for (int j = 0; j < nTriangles; j++)
//                {
//                    bvh8Node.meta[i] |= 1 << (j + 5);
//                }
//                // Low 5 bits store the index of first triangle relative to the triangle base index
//                bvh8Node.meta[i] |= nTrianglesTotal;
//
//                nTrianglesTotal += nTriangles;
//            }
//        }
//    }
//	assert(nTrianglesTotal <= 24);
//
//
//    // Caching child base index before resizing nodes array
//    uint32_t childBaseIdx = bvh8Node.childBaseIdx;
//
//    bvh8.nodes.resize(m_UsedNodes);
//
//    int childCount = 0;
//    // Recursively collapse internal children nodes
//    for (int i = 0; i < 8; i++)
//    {
//        if (childrenIndices[i] == -1)
//            continue;
//
//        const NodeEval& eval = m_Evals[childrenIndices[i]][0];
//
//        if (eval.decision == Decision::INTERNAL)
//        {
//            CollapseNode(bvh8, childrenIndices[i], childBaseIdx + childCount);
//            childCount++;
//        }
//    }
//}
