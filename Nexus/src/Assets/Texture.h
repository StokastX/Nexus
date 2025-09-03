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
		METALNESS,
		METALLICROUGHNESS,
		EMISSIVE,
		NORMALS
	};

	Texture() = default;
	Texture(uint32_t w, uint32_t h, uint32_t c, bool isHDR, void* d);
	Texture(const Texture& other);
	Texture(Texture&& other);
	Texture& operator=(const Texture& other);
	Texture& operator=(Texture&& other);
	~Texture();

	static cudaTextureObject_t ToDevice(const Texture& texture);
	static void DestructFromDevice(const cudaTextureObject_t& texture);

	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t channels = 0;
	bool sRGB = true;
	bool HDR = false;

	void* pixels = nullptr;
	Type type = Type::DIFFUSE;

private:
	void CopyFrom(const Texture& other);
};