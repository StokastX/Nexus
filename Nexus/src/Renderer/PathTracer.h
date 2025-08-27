#pragma once
#include "Device/Kernels/CUDAKernel.h"
#include "OpenGL/PixelBuffer.h"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Device/DeviceVector.h"
#include "Device/Kernels/CUDAGraph.h"


class PathTracer
{
public:
	PathTracer(uint2 resolution);
	~PathTracer();

	void FreeDeviceBuffers();
	void Reset();
	void ResetFrameNumber();
	void Render(const Scene& scene);
	void OnResize(uint2 resolution);

	void UpdateDeviceScene(const Scene& scene);

	void SetPixelQuery(uint32_t x, uint32_t y);
	bool PixelQueryPending() { return m_PixelQueryPending; }

	int32_t GetSelectedInstance() { return m_PixelQuery->instanceIdx; }
	int32_t SynchronizePixelQuery();
	uint32_t GetFrameNumber() const { return m_FrameNumber; }
	const PixelBuffer& GetPixelBuffer() { return m_PixelBuffer; }

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
	DeviceInstance<float3*> m_AccumulationBuffer;
	DeviceInstance<uint32_t> m_DeviceFrameNumber;
	DeviceInstance<uint32_t> m_DeviceBounce;
	DeviceInstance<uint32_t*> m_RenderBuffer;

	PixelBuffer m_PixelBuffer;

	DeviceInstance<Scene, D_Scene> m_Scene;

	DeviceInstance<D_PixelQuery> m_PixelQuery;
	bool m_PixelQueryPending = false;

	DeviceInstance<D_PathStateSOA> m_PathState;

	DeviceInstance<D_TraceRequestSOA> m_TraceRequest;
	DeviceInstance<D_ShadowTraceRequestSOA> m_ShadowTraceRequest;

	DeviceInstance<D_MaterialRequestSOA> m_MaterialRequest;
	DeviceInstance<D_QueueSize> m_QueueSize;
};