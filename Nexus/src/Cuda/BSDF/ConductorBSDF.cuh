#pragma once

#include "Cuda/Random.cuh"
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Geometry/Ray.cuh"
#include "Utils/cuda_math.h"
#include "Microfacet.cuh"
#include "Fresnel.cuh"

struct D_ConductorBSDF
{
	float alpha;
	float3 eta;
	float3 k;

	inline __device__ void PrepareBSDFData(const float3& wi, const D_Material& material)
	{
		alpha = clamp((1.2f - 0.2f * sqrtf(fabs(wi.z))) * material.conductor.roughness * material.conductor.roughness, 1.0e-4f, 1.0f);
		eta = wi.z < 0.0f ? material.conductor.ior : 1 / material.conductor.ior;
		k = material.conductor.k;
	}

	inline __device__ bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{
		const float3 m = Microfacet::SampleSpecularHalfBeckWalt(alpha, rngState);

		const float wiDotM = dot(wi, m);

		float3 F82 = material.principled.specularColor * Fresnel::SchlickMetallicReflectance(material.principled.albedo, wiDotM);
		float3 F = material.principled.specularWeight * Fresnel::SchlickMetallicF82(material.principled.albedo, F82, wiDotM);
		//float3 F = Fresnel::ComplexReflectance(wiDotM, material.conductor.ior, k);

		wo = reflect(-wi, m);

		const float weight = Microfacet::WeightBeckmannWalter(alpha, abs(wiDotM), abs(wo.z), abs(wi.z), m.z);

		// If the new ray is under the hemisphere, return
		if (wo.z * wi.z < 0.0f)
			return false;

		throughput = weight * F;
		pdf = Microfacet::SampleWalterReflectionPdf(alpha, m.z, fabs(wiDotM));

		return true;
	}
};