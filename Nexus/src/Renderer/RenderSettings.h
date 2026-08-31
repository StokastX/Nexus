#pragma once
#include "Utils/cuda_math.h"
#include "Utils/ColorUtils.h"
#include "Device/DeviceTraits.h"
#include "Cuda/Scene/Scene.cuh"

namespace Nexus {

	struct RenderSettings
	{
		uint2 resolution;

		bool useMIS = true;
		bool visualizeBvh = false;
		bool wireframeBvh = false;
		unsigned char pathLength = 10;

		float3 backgroundColor = make_float3(0.034f);
		float backgroundIntensity = 1.0f;
		ColorUtils::ToneMapping toneMapping = ColorUtils::ToneMapping::AGX_DEFAULT;
		float exposure = 0.0f;
	};


	template<>
	struct DeviceTraits<RenderSettings>
	{
		using DeviceType = D_RenderSettings;

		static D_RenderSettings ToDevice(const RenderSettings& settings)
		{
			D_RenderSettings deviceSettings;
			deviceSettings.resolution = settings.resolution;
			deviceSettings.useMIS = settings.useMIS;
			deviceSettings.visualizeBvh = settings.visualizeBvh;
			deviceSettings.wireframeBvh = settings.wireframeBvh;
			deviceSettings.pathLength = settings.pathLength;
			deviceSettings.backgroundColor = settings.backgroundColor;
			deviceSettings.backgroundIntensity = settings.backgroundIntensity;
			deviceSettings.toneMapping = settings.toneMapping;
			deviceSettings.exposure = settings.exposure;
			return deviceSettings;
		}
	};

	static_assert(sizeof(RenderSettings) == sizeof(D_RenderSettings),
		"RenderSettings and D_RenderSettings have diverged -- update DeviceTraits<RenderSettings>::ToDevice");

}
