#pragma once
#include <string>
#include "Cuda/Scene/Material.cuh"
#include "Utils/cuda_math.h"
#include "Device/DeviceTraits.h"

namespace Nexus {

	struct Material
	{
		float3 baseColor = make_float3(0.8f);
		float metalness = 0.0f;
		float roughness = 0.3f;
		float anisotropy = 0.0f;
		float specularWeight = 1.0f;
		float3 specularColor = make_float3(1.0f);
		float ior = 1.5f;
		float transmission = 0.0f;

		float3 emissionColor = make_float3(1.0f);
		float intensity = 0.0f;
		float opacity = 1.0f;

		int32_t baseColorMapId = -1;
		int32_t emissiveMapId = -1;
		int32_t normalMapId = -1;
		int32_t roughnessMapId = -1;
		int32_t metalnessMapId = -1;
		int32_t metallicRoughnessMapId = -1;
	};


	template<>
	struct DeviceTraits<Material>
	{
		using DeviceType = D_Material;

		static D_Material ToDevice(const Material& material)
		{
			D_Material deviceMaterial;
			deviceMaterial.baseColor = material.baseColor;
			deviceMaterial.metalness = material.metalness;
			deviceMaterial.roughness = material.roughness;
			deviceMaterial.anisotropy = material.anisotropy;
			deviceMaterial.specularWeight = material.specularWeight;
			deviceMaterial.specularColor = material.specularColor;
			deviceMaterial.ior = material.ior;
			deviceMaterial.transmission = material.transmission;
			deviceMaterial.emissionColor = material.emissionColor;
			deviceMaterial.intensity = material.intensity;
			deviceMaterial.opacity = material.opacity;
			deviceMaterial.baseColorMapId = material.baseColorMapId;
			deviceMaterial.emissiveMapId = material.emissiveMapId;
			deviceMaterial.normalMapId = material.normalMapId;
			deviceMaterial.roughnessMapId = material.roughnessMapId;
			deviceMaterial.metalnessMapId = material.metalnessMapId;
			deviceMaterial.metallicRoughnessMapId = material.metallicRoughnessMapId;
			return deviceMaterial;
		}
	};

	// The two are meant to stay field-for-field identical; this catches a field added to one and
	// not the other. Drop it the day they are deliberately allowed to diverge.
	static_assert(sizeof(Material) == sizeof(D_Material),
		"Material and D_Material have diverged -- update DeviceTraits<Material>::ToDevice");

}
