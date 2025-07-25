#pragma once

#include <cuda_runtime_api.h>
#include "Material.cuh"
#include "Camera.cuh"
#include "Light.cuh"

#include "Renderer/RenderSettings.h"

struct D_RenderSettings
{
	bool useMIS;
	bool visualizeBvh;
	bool wireFrameBvh;
	unsigned char pathLength;

	float3 backgroundColor;
	float backgroundIntensity;
};

struct D_Scene
{
	bool hasHdrMap;
	cudaTextureObject_t hdrMap;

	cudaTextureObject_t* textures;

	//D_TLAS tlas;

	D_Light* lights;
	uint32_t lightCount;

	D_Material* materials;
	D_Camera camera;

	D_MeshInstance* meshInstances;

	D_RenderSettings renderSettings;
};