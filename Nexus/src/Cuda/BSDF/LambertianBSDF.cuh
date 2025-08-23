#pragma once
#include "Cuda/Random.cuh"
#include "Cuda/Geometry/Ray.cuh"
#include "Cuda/Scene/Material.cuh"
#include "Utils/Utils.h"
#include "Cuda/Sampler.cuh"

// Basic lambertian (diffuse) BSDF
struct D_LambertianBSDF
{
	inline __device__ void PrepareBSDFData(const float3& wi, const D_Material& material)
	{

	}

	inline __device__ bool Eval(const D_Material& material, const float3& wi, const float3& wo, float3& throughput, float& pdf)
	{
		const float wiDotN = wi.z;
		const float woDotN = wo.z;
		const bool reflected = wiDotN * woDotN > 0.0f;

		if (!reflected)
			return false;

		throughput = material.baseColor * INV_PI * woDotN;
		pdf = INV_PI * woDotN;

		return Sampler::IsPdfValid(pdf);
	}

	inline __device__ bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{
		wo = Random::RandomCosineHemisphere(rngState);
		throughput = material.baseColor;
		pdf = INV_PI * wo.z;

		return Sampler::IsPdfValid(pdf);
	}
};