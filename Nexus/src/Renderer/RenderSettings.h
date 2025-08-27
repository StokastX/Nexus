#pragma once
#include "Utils/cuda_math.h"
#include "Utils/ColorUtils.h"

struct RenderSettings
{
	uint2 resolution;

	bool useMIS = true;
	bool visualizeBvh = false;
	bool wireframeBvh = false;
	unsigned char pathLength = 10;

	float3 backgroundColor = make_float3(0.0f);
	float backgroundIntensity = 1.0f;
	ColorUtils::ToneMapping toneMapping = ColorUtils::ToneMapping::AGX_DEFAULT;
	float exposure = 0.0f;
};