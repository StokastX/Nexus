#pragma once
#include <iostream>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <Utils/cuda_math.h>

struct Texture
{
	enum struct Type {
		DIFFUSE,
		ROUGHNESS,
		METALLIC,
		EMISSIVE,
		NORMALS
	};

	Texture() = default;
	Texture(uint32_t w, uint32_t h, uint32_t c, unsigned char* d);

	static cudaTextureObject_t ToDevice(const Texture& texture);
	static void DestructFromDevice(const cudaTextureObject_t& texture);

	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t channels = 0;
	bool sRGB = true;

	unsigned char* pixels = nullptr;
	Type type = Type::DIFFUSE;
};