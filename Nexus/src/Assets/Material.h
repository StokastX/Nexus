#pragma once
#include <string>
#include "Cuda/Scene/Material.cuh"
#include "Utils/cuda_math.h"

struct Material
{
	float3 baseColor = make_float3(0.8f);
	float metalness = 0.0f;
	float roughness = 0.3f;
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

	//static D_Material ToDevice(const Material& material)
	//{
	//	D_Material deviceMaterial;
	//	memcpy(&deviceMaterial, &material, sizeof(D_Material));
	//	return deviceMaterial;
	//}

	static std::string GetMaterialTypesString()
	{
		std::string materialTypes;
		materialTypes.append("Diffuse");
		materialTypes.push_back('\0');
		materialTypes.append("Dielectric");
		materialTypes.push_back('\0');
		materialTypes.append("Plastic");
		materialTypes.push_back('\0');
		materialTypes.append("Conductor");
		materialTypes.push_back('\0');
		return materialTypes;
	}
};

