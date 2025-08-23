#pragma once
#include <cstdint>

static const char *lightTypeNames[] = {
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
