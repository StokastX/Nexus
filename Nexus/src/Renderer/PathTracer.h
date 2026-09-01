#pragma once
#include "Device/Kernels/CUDAKernel.h"
#include "OpenGL/InteropTexture.h"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Device/DeviceVector.h"
#include "Device/DeviceSymbol.h"
#include "Scene/Scene.h"
#include "Device/Kernels/CUDAGraph.h"


namespace Nexus {

	class PathTracer
	{
	public:
		PathTracer(uint2 resolution);
		~PathTracer();

		void FreeDeviceBuffers();
		void Reset();
		void ResetFrameNumber();
		// `target` is the display texture the accumulate kernel writes into. It is mapped for
		// CUDA for the duration of the call and handed back to OpenGL before returning.
		void Render(const Scene& scene, InteropTexture& target);
		void OnResize(uint2 resolution);

		void UpdateDeviceScene(const Scene& scene);

		void SetPixelQuery(uint32_t x, uint32_t y);
		bool PixelQueryPending() { return m_PixelQueryPending; }

		int32_t GetSelectedInstance() { return m_PixelQuery->instanceIdx; }
		int32_t SynchronizePixelQuery();
		uint32_t GetFrameNumber() const { return m_FrameNumber; }

	private:
		CUDAKernel m_GenerateKernel;
		CUDAKernel m_LogicKernel;
		CUDAKernel m_TraceKernel;
		CUDAKernel m_TraceVisualizeBvhKernel;
		CUDAKernel m_TraceShadowKernel;
		CUDAKernel m_MaterialKernel;
		CUDAKernel m_AccumulateKernel;

		CUDAGraph m_RenderGraph;


		uint32_t m_FrameNumber = 0;

		uint2 m_Resolution;

		// Device members
		DeviceSymbol<float3*> m_AccumulationBuffer;
		DeviceSymbol<uint32_t> m_DeviceFrameNumber;
		DeviceSymbol<uint32_t> m_DeviceBounce;
		DeviceSymbol<cudaSurfaceObject_t> m_RenderSurface;

		DeviceSymbol<Scene> m_Scene;

		DeviceSymbol<D_PixelQuery> m_PixelQuery;
		bool m_PixelQueryPending = false;

		DeviceSymbol<D_PathStateSOA> m_PathState;

		DeviceSymbol<D_TraceRequestSOA> m_TraceRequest;
		DeviceSymbol<D_ShadowTraceRequestSOA> m_ShadowTraceRequest;

		DeviceSymbol<D_MaterialRequestSOA> m_MaterialRequest;
		DeviceSymbol<D_QueueSize> m_QueueSize;
	};

}