#include "CUDAGraph.h"

#include "Utils/Utils.h"

namespace Nexus {

	CUDAGraph::CUDAGraph()
	{
		CheckCudaErrors(cudaGraphCreate(&m_Graph, 0));
		CheckCudaErrors(cudaStreamCreate(&m_Stream));
		BuildGraph();
	}

	CUDAGraph::~CUDAGraph()
	{
		CheckCudaErrors(cudaGraphExecDestroy(m_GraphExec));
		CheckCudaErrors(cudaStreamDestroy(m_Stream));
		CheckCudaErrors(cudaGraphDestroy(m_Graph));
	}

	void CUDAGraph::Reset()
	{
		CheckCudaErrors(cudaGraphDestroy(m_Graph));
		CheckCudaErrors(cudaGraphCreate(&m_Graph, 0));
	}

	void CUDAGraph::BuildGraph()
	{
		CheckCudaErrors(cudaGraphInstantiate(&m_GraphExec, m_Graph, 0));
	}

	void CUDAGraph::Execute()
	{
		CheckCudaErrors(cudaGraphLaunch(m_GraphExec, m_Stream));
	}

	cudaGraphNode_t CUDAGraph::AddKernelNode(CUDAKernel& kernel, cudaGraphNode_t* dependencies, size_t numDependencies)
	{
		cudaGraphNode_t kernelNode;
		cudaKernelNodeParams nodeParams;
		nodeParams.blockDim = kernel.GetBlockSize();
		nodeParams.gridDim = kernel.GetGridSize();
		nodeParams.extra = nullptr;
		nodeParams.kernelParams = kernel.GetLaunchParameters();
		nodeParams.func = kernel.GetFunction();
		nodeParams.sharedMemBytes = 0;

		CheckCudaErrors(cudaGraphAddKernelNode(&kernelNode, m_Graph, dependencies, numDependencies, &nodeParams));
		return kernelNode;
	}

}
