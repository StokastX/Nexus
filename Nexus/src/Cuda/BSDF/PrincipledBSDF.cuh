#pragma once

#include "PlasticBSDF.cuh"
#include "DielectricBSDF.cuh"
#include "ConductorBSDF.cuh"

class PrincipledBSDF
{
	inline __device__ bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{

	}
};