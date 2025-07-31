#include "PathTracer.h"
#include "Device/CudaMemory.h"


PathTracer::PathTracer(uint32_t width, uint32_t height)
	: m_ViewportWidth(width),
	m_ViewportHeight(height),
	m_PixelBuffer(width, height),
	m_AccumulationBuffer(GetDeviceAccumulationBufferAddress()),
	m_RenderBuffer(GetDeviceRenderBufferAddress()),
	m_Scene(GetDeviceSceneAddress()),
	m_DeviceFrameNumber(GetDeviceFrameNumberAddress()),
	m_DeviceBounce(GetDeviceBounceAddress()),
	m_PathState(GetDevicePathStateAddress()),
	m_ShadowTraceRequest(GetDeviceShadowTraceRequestAddress()),
	m_TraceRequest(GetDeviceTraceRequestAddress()),
	m_MaterialRequest(GetDeviceMaterialRequestAddress()),
	m_QueueSize(GetDeviceQueueSizeAddress()),
	m_PixelQuery(GetDevicePixelQueryAddress())
{
	Reset();
}

PathTracer::~PathTracer()
{
	FreeDeviceBuffers();
	CheckCudaErrors(cudaDeviceSynchronize());
}

void PathTracer::FreeDeviceBuffers()
{
	CudaMemory::FreeAsync(m_AccumulationBuffer.Instance());
	CudaMemory::FreeAsync(m_PathState->lastPdf);
	CudaMemory::FreeAsync(m_PathState->throughput);
	CudaMemory::FreeAsync(m_PathState->radiance);
	CudaMemory::FreeAsync(m_PathState->allowMIS);

	CudaMemory::FreeAsync(m_TraceRequest->intersection.hitDistance);
	CudaMemory::FreeAsync(m_TraceRequest->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_TraceRequest->intersection.triIdx);
	CudaMemory::FreeAsync(m_TraceRequest->intersection.u);
	CudaMemory::FreeAsync(m_TraceRequest->intersection.v);
	CudaMemory::FreeAsync(m_TraceRequest->pixelIdx);
	CudaMemory::FreeAsync(m_TraceRequest->ray.origin);
	CudaMemory::FreeAsync(m_TraceRequest->ray.direction);

	CudaMemory::FreeAsync(m_ShadowTraceRequest->hitDistance);
	CudaMemory::FreeAsync(m_ShadowTraceRequest->pixelIdx);
	CudaMemory::FreeAsync(m_ShadowTraceRequest->radiance);
	CudaMemory::FreeAsync(m_ShadowTraceRequest->ray.origin);
	CudaMemory::FreeAsync(m_ShadowTraceRequest->ray.direction);

	CudaMemory::FreeAsync(m_MaterialRequest->intersection.hitDistance);
	CudaMemory::FreeAsync(m_MaterialRequest->intersection.instanceIdx);
	CudaMemory::FreeAsync(m_MaterialRequest->intersection.triIdx);
	CudaMemory::FreeAsync(m_MaterialRequest->intersection.u);
	CudaMemory::FreeAsync(m_MaterialRequest->intersection.v);
	CudaMemory::FreeAsync(m_MaterialRequest->rayDirection);
	CudaMemory::FreeAsync(m_MaterialRequest->pixelIdx);
}

void PathTracer::Reset()
{
	m_AccumulationBuffer = CudaMemory::Allocate<float3>(m_ViewportWidth * m_ViewportHeight);

	dim3 gridSize(m_ViewportWidth * m_ViewportHeight / BLOCK_SIZE + 1, 1, 1);
	dim3 blockSize(BLOCK_SIZE, 1, 1);

	m_GenerateKernel = CUDAKernel((void*)GenerateKernel, gridSize, blockSize);
	m_LogicKernel = CUDAKernel((void*)LogicKernel, gridSize, blockSize);
	m_MaterialKernel = CUDAKernel((void*)MaterialKernel, gridSize, blockSize);
	m_TraceKernel = CUDAKernel((void*)TraceKernel);
	m_TraceShadowKernel = CUDAKernel((void*)TraceShadowKernel);
	m_AccumulateKernel = CUDAKernel((void*)AccumulateKernel, gridSize, blockSize);

	// Set minimal launch configuration for trace kernels.
	// Inactive threads will fetch new rays in the trace queue.
	m_TraceKernel.SetMinimalLaunchConfigurationWithBlockSize(BLOCK_SIZE);
	m_TraceShadowKernel.SetMinimalLaunchConfigurationWithBlockSize(BLOCK_SIZE);

	m_RenderGraph.Reset();
	cudaGraphNode_t logicNode = m_RenderGraph.AddKernelNode(m_LogicKernel);
	cudaGraphNode_t materialNode = m_RenderGraph.AddKernelNode(m_MaterialKernel, &logicNode, 1);
	cudaGraphNode_t traceNode = m_RenderGraph.AddKernelNode(m_TraceKernel, &materialNode, 1);
	cudaGraphNode_t traceShadowNode = m_RenderGraph.AddKernelNode(m_TraceShadowKernel, &materialNode, 1);

	m_RenderGraph.BuildGraph();

	const uint32_t count = m_ViewportWidth * m_ViewportHeight;

	float* lastPdf = CudaMemory::AllocateAsync<float>(count);
	float3* throughput = CudaMemory::AllocateAsync<float3>(count);
	float3* rayOrigin = CudaMemory::AllocateAsync<float3>(count);
	float3* radiance = CudaMemory::AllocateAsync<float3>(count);
	bool* allowMIS = CudaMemory::AllocateAsync<bool>(count);

	D_PathStateSOA pathState;
	pathState.lastPdf = lastPdf;
	pathState.throughput = throughput;
	pathState.radiance = radiance;
	pathState.allowMIS = allowMIS;

	m_PathState = pathState;

	D_IntersectionSOA intersectionSOA;
	intersectionSOA.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSOA.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSOA.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSOA.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSOA.v = CudaMemory::AllocateAsync<float>(count);

	D_RaySOA raySOA;
	raySOA.origin = CudaMemory::AllocateAsync<float3>(count);
	raySOA.direction = CudaMemory::AllocateAsync<float3>(count);

	uint32_t* pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);

	D_TraceRequestSOA traceRequest;

	traceRequest.intersection = intersectionSOA;
	traceRequest.ray = raySOA;
	traceRequest.pixelIdx = pixelIdx;

	m_TraceRequest = traceRequest;

	float* hitDistance = CudaMemory::AllocateAsync<float>(count);
	pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	radiance = CudaMemory::AllocateAsync<float3>(count);
	raySOA.origin = CudaMemory::AllocateAsync<float3>(count);
	raySOA.direction = CudaMemory::AllocateAsync<float3>(count);

	D_ShadowTraceRequestSOA shadowTraceRequest;
	shadowTraceRequest.hitDistance = hitDistance;
	shadowTraceRequest.pixelIdx = pixelIdx;
	shadowTraceRequest.radiance = radiance;
	shadowTraceRequest.ray = raySOA;

	m_ShadowTraceRequest = shadowTraceRequest;

	intersectionSOA.hitDistance = CudaMemory::AllocateAsync<float>(count);
	intersectionSOA.instanceIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSOA.triIdx = CudaMemory::AllocateAsync<uint32_t>(count);
	intersectionSOA.u = CudaMemory::AllocateAsync<float>(count);
	intersectionSOA.v = CudaMemory::AllocateAsync<float>(count);
	float3* rayDirection = CudaMemory::AllocateAsync<float3>(count);
	pixelIdx = CudaMemory::AllocateAsync<uint32_t>(count);

	D_MaterialRequestSOA materialRequest;
	materialRequest.intersection = intersectionSOA;
	materialRequest.rayDirection = rayDirection;
	materialRequest.pixelIdx = pixelIdx;

	m_MaterialRequest = materialRequest;

	// Set pixel query index to undefined
	D_PixelQuery pixelQuery;
	pixelQuery.pixelIdx = -1;
	pixelQuery.instanceIdx = -1;

	m_PixelQuery = pixelQuery;
}

void PathTracer::ResetFrameNumber()
{
	m_FrameNumber = 0;
}

void PathTracer::Render(const Scene& scene)
{
	m_FrameNumber++;

	CheckCudaErrors(cudaGraphicsMapResources(1, &m_PixelBuffer.GetCudaResource()));
	size_t size = 0;
	uint32_t* devicePtr = 0;
	CheckCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, m_PixelBuffer.GetCudaResource()));

	m_RenderBuffer = devicePtr;

	m_DeviceFrameNumber = m_FrameNumber;
	m_DeviceBounce = 0;

	// Reset queue sizes
	CudaMemory::MemsetAsync(m_QueueSize.Data(), 0, sizeof(D_QueueSize));

	// Primary rays
	m_GenerateKernel.Launch();
	m_TraceKernel.Launch();

	m_DeviceBounce = 1;

	// Secondary rays
	for (uint32_t i = 0; i < scene.GetRenderSettings().pathLength; i++)
	{
		m_RenderGraph.Execute();
		m_DeviceBounce = i + 2;
	}

	m_AccumulateKernel.Launch();

	if (m_PixelQueryPending)
	{
		m_PixelQueryPending = false;
		m_PixelQuery.Synchronize();
		D_PixelQuery pixelQuery;
		pixelQuery.pixelIdx = -1;
		pixelQuery.instanceIdx = m_PixelQuery->instanceIdx;
		m_PixelQuery = pixelQuery;
	}

	CheckCudaErrors(cudaGetLastError());
	CheckCudaErrors(cudaGraphicsUnmapResources(1, &m_PixelBuffer.GetCudaResource(), 0));
}

void PathTracer::OnResize(uint32_t width, uint32_t height)
{
	if ((m_ViewportWidth != width || m_ViewportHeight != height) && width != 0 && height != 0)
	{
		m_FrameNumber = 0;
		m_PixelBuffer.OnResize(width, height);

		m_ViewportWidth = width;
		m_ViewportHeight = height;

		FreeDeviceBuffers();
		Reset();
	}
}

void PathTracer::UpdateDeviceScene(const Scene& scene)
{
	m_Scene = scene;
}

void PathTracer::SetPixelQuery(uint32_t x, uint32_t y)
{
	D_PixelQuery pixelQuery;
	pixelQuery.pixelIdx = m_ViewportWidth * y + x;
	pixelQuery.instanceIdx = -1;
	m_PixelQuery = pixelQuery;
	m_PixelQueryPending = true;
}

