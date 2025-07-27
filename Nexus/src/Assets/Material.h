#pragma once
#include <string>
#include "Cuda/Scene/Material.cuh"
#include "Utils/cuda_math.h"

struct Material
{
	enum struct Type : char
	{
		DIFFUSE,
		DIELECTRIC,
		PLASTIC,
		CONDUCTOR
	};

	union
	{
		struct
		{
			float3 albedo;
		} diffuse;

		struct
		{
			float3 albedo;
			float roughness;
			float ior;
		} dielectric;

		struct
		{
			float3 albedo;
			float roughness;
			float ior;
		} plastic;

		struct
		{
			float3 ior;
			float3 k;
			float roughness;
		} conductor;
	};

	float3 emissive;
	float intensity;
	float opacity = 1.0f;

	int diffuseMapId = -1;
	int emissiveMapId = -1;
	int normalMapId = -1;
	int roughnessMapId = -1;
	Type type;

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

