#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "Utils/Utils.h"

namespace Nexus {

	class CUDAKernel
	{
	public:
		CUDAKernel() = default;
		CUDAKernel(void* function, dim3 gridsize = dim3(1, 0, 0), dim3 blockSize = dim3(1, 0, 0), std::vector<void*> params = std::vector<void*>())
			: m_Function(function), m_GridSize(gridsize), m_BlockSize(blockSize), m_LaunchParameters(params) {
		}

		virtual void Init() {}
		virtual void Launch()
		{
			CheckCudaErrors(cudaLaunchKernel(m_Function, m_GridSize, m_BlockSize, m_LaunchParameters.data(), 0, 0));
		}

		const dim3 GetBlockSize() const { return m_BlockSize; }
		const dim3 GetGridSize() const { return m_GridSize; }
		void** GetLaunchParameters() { return m_LaunchParameters.data(); }
		void* GetFunction() const { return m_Function; }

		void SetLaunchConfiguration(dim3 GridSize, dim3 BlockSize)
		{
			m_GridSize = GridSize;
			m_BlockSize = BlockSize;
		}

		void SetMinimalLaunchConfigurationWithBlockSize(int32_t blockSize)
		{
			int device;
			cudaDeviceProp prop;

			cudaGetDevice(&device);
			cudaGetDeviceProperties(&prop, device);

			int gridSize;
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&gridSize, m_Function, blockSize, 0);
			m_GridSize = dim3(gridSize * prop.multiProcessorCount, 1, 1);
			m_BlockSize = dim3(blockSize, 1, 1);

			//cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, m_Function);
		}

		// Minimal launch configuration suggested to achieve full gpu occupancy
		void SetMinimalLaunchConfiguration()
		{
			int gridSize, blockSize;
			cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, m_Function);
			m_GridSize = dim3(gridSize, 1, 1);
			m_BlockSize = dim3(blockSize, 1, 1);
		}

		void SetLaunchParameters(const std::vector<void*>& params)
		{
			m_LaunchParameters = params;
		}

	private:

		void* m_Function = nullptr;
		std::vector<void*> m_LaunchParameters;

		dim3 m_GridSize = dim3(1, 1, 1);
		dim3 m_BlockSize = dim3(1, 1, 1);
	};

}
