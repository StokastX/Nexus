#pragma once
#include "Scene/Scene.h"
#include "Cuda/PathTracer/PathTracer.cuh"
#include "Renderer/Renderer.h"
#include <imgui.h>

class MetricsPanel 
{
public:
	MetricsPanel(Renderer* context);

	void Reset();
	void UpdateMetrics(float deltaTime);
	void OnImGuiRender(uint32_t frameNumber, ImVec2 viewportSize);
	bool FitRenderToViewport() { return m_FitRenderToViewport; }

private:

	Renderer* m_Context;

	bool m_FitRenderToViewport = true;
	uint32_t m_NAccumulatedFrame = 0;
	float m_AccumulatedTime = 0.0f;
	float m_DisplayFPSTimer = 0.0f;
	float m_DeltaTime = 0.0f;

	float m_MRPS = 0.0f;
	uint32_t m_NumRaysProcessed = 0;

};