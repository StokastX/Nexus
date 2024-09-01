#pragma once

#include <vector>
#include <cuda_runtime_api.h>
#include "CUDAKernel.h"

class CUDAGraph
{
public:
	CUDAGraph();
	~CUDAGraph();

	void Reset();
	void BuildGraph();
	void Execute();
	void AddKernelNode(CUDAKernel& kernel);

private:
	cudaGraph_t m_Graph;
	cudaGraphExec_t m_GraphExec;
	cudaStream_t m_Stream;

	std::vector<cudaGraphNode_t> m_Nodes;

};