#pragma once
#include "Utils/cuda_math.h"

struct RenderSettings
{
	bool useMIS = true;
	bool visualizeBvh = false;
	bool wireframeBvh = false;
	unsigned char pathLength = 10;

	float3 backgroundColor = make_float3(0.0f);
	float backgroundIntensity = 1.0f;
};