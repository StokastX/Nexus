#pragma once
#include <cstdint>
#include "Device/DeviceTraits.h"
#include "Cuda/Scene/Light.cuh"

namespace Nexus {

	static const char* lightTypeNames[] = {
		"Point Light",
		"Spot Light",
		"Directional Light"
	};

	struct Light
	{
		enum struct Type : char
		{
			POINT,
			SPOT,
			DIRECTIONAL,
			MESH,
			UNDEFINED
		};

		union
		{
			struct
			{
				float3 position;
				float3 color;
				float intensity;
			} point;

			struct
			{
				float3 position;
				float3 direction;
				float3 color;
				float intensity;
				float falloffStart;
				float falloffEnd;
			} spot;

			struct
			{
				float3 color;
				float3 direction;
				float intensity;
			} directional;

			struct
			{
				uint32_t meshId;
			} mesh;
		};

		Type type = Type::UNDEFINED;
	};


	template<>
	struct DeviceTraits<Light>
	{
		using DeviceType = D_Light;

		// Copies through the active union member only -- a blanket field copy would read whichever
		// alternative happens to be largest, regardless of which one was written.
		static D_Light ToDevice(const Light& light)
		{
			D_Light deviceLight;
			deviceLight.type = static_cast<D_Light::Type>(light.type);

			switch (light.type)
			{
			case Light::Type::POINT:
				deviceLight.point.position = light.point.position;
				deviceLight.point.color = light.point.color;
				deviceLight.point.intensity = light.point.intensity;
				break;

			case Light::Type::SPOT:
				deviceLight.spot.position = light.spot.position;
				deviceLight.spot.direction = light.spot.direction;
				deviceLight.spot.color = light.spot.color;
				deviceLight.spot.intensity = light.spot.intensity;
				deviceLight.spot.falloffStart = light.spot.falloffStart;
				deviceLight.spot.falloffEnd = light.spot.falloffEnd;
				break;

			case Light::Type::DIRECTIONAL:
				deviceLight.directional.color = light.directional.color;
				deviceLight.directional.direction = light.directional.direction;
				deviceLight.directional.intensity = light.directional.intensity;
				break;

			case Light::Type::MESH:
				deviceLight.mesh.meshId = light.mesh.meshId;
				break;

			case Light::Type::UNDEFINED:
				break;
			}

			return deviceLight;
		}
	};

	static_assert(sizeof(Light) == sizeof(D_Light),
		"Light and D_Light have diverged -- update DeviceTraits<Light>::ToDevice");

}
