#pragma once

#include <vector>
#include <cuda_runtime_api.h>
#include "CUDAKernel.h"

namespace Nexus {

	class CUDAGraph
	{
	public:
		CUDAGraph();
		~CUDAGraph();

		void Reset();
		void BuildGraph();
		void Execute();
		cudaGraphNode_t AddKernelNode(CUDAKernel& kernel, cudaGraphNode_t* dependencies = nullptr, size_t numDependencies = 0);

	private:
		cudaGraph_t m_Graph;
		cudaGraphExec_t m_GraphExec;
		cudaStream_t m_Stream;

		std::vector<cudaGraphNode_t> m_Nodes;

	};

}
