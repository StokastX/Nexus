#pragma once

#include "Utils/cuda_math.h"

struct D_Material
{
	enum struct D_Type : char
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
	float opacity;

	int diffuseMapId = -1;
	int emissiveMapId = -1;
	int normalMapId = -1;
	int roughnessMapId = -1;
	D_Type type;
};