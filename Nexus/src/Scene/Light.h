#pragma once
#include <cstdint>

struct Light
{
	enum struct Type : char
	{
		UNDEFINED,
		POINT,
		SPOT,
		DIRECTIONAL,
		MESH
	};

	union
	{
		struct
		{
			float3 color;
			float intensity;
		} point;

		struct
		{
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
