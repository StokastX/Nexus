#pragma once

#include <cuda_runtime_api.h>
#include "Material.cuh"
#include "Camera.cuh"
#include "Light.cuh"

#include "Renderer/RenderSettings.h"

struct D_RenderSettings
{
	bool useMIS;
	unsigned char pathLength;

	float3 backgroundColor;
	float backgroundIntensity;
};

struct D_Scene
{
	bool hasHdrMap;
	cudaTextureObject_t hdrMap;

	cudaTextureObject_t* diffuseMaps;
	cudaTextureObject_t* emissiveMaps;

	//D_TLAS tlas;

	D_Light* lights;
	uint32_t lightCount;

	D_Material* materials;
	D_Camera camera;

	D_MeshInstance* meshInstances;

	D_RenderSettings renderSettings;
};