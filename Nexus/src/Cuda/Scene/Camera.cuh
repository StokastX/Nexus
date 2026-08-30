#pragma once

#include "Utils/cuda_math.h"

namespace Nexus {

	struct D_Camera
	{
		float3 position;
		float3 right;
		float3 up;
		float lensRadius;
		float3 lowerLeftCorner;
		float3 viewportX;
		float3 viewportY;
		uint2 resolution;
	};

}