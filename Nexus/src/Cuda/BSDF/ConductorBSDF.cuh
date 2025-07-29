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

	inline __device__ void PrepareBSDFData(const float3& wi, const D_Material& material)
	{
		alpha = clamp((1.2f - 0.2f * sqrtf(fabs(wi.z))) * material.roughness * material.roughness, 1.0e-4f, 1.0f);
	}

	inline __device__ bool Eval(const D_Material& material, const float3& wi, const float3& wo, float3& throughput, float& pdf)
	{
		const float wiDotN = wi.z;
		const float woDotN = wo.z;

		const float3 m = normalize(wo + wi);
		float cosThetaT;
		const float wiDotM = dot(wi, m);
		float3 F82 = material.specularColor * Fresnel::SchlickMetallicReflectance(material.baseColor, fabs(wiDotM));
		float3 F = material.specularWeight * Fresnel::SchlickMetallicF82(material.baseColor, F82, fabs(wiDotM));
		const float G = Microfacet::Smith_G2(alpha, fabs(woDotN), fabs(wiDotN));
		const float D = Microfacet::BeckmannD(alpha, m.z);

		// BRDF times woDotN
		throughput = F * G * D / (4.0f * fabs(wiDotN));

		// pm * jacobian = pm * || dWhr / dWo ||
		// We can replace woDotM with wiDotM since for a reflection both are equal
		pdf = D * fabs(m.z) / (4.0f * fabs(wiDotM));

		return Sampler::IsPdfValid(pdf);
	}

	inline __device__ bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{
		const float3 m = Microfacet::SampleSpecularHalfBeckWalt(alpha, rngState);

		const float wiDotM = dot(wi, m);

		float3 F82 = material.specularColor * Fresnel::SchlickMetallicReflectance(material.baseColor, wiDotM);
		float3 F = material.specularWeight * Fresnel::SchlickMetallicF82(material.baseColor, F82, wiDotM);

		wo = reflect(-wi, m);

		// If the new ray is under the hemisphere, return
		if (wo.z * wi.z < 0.0f)
			return false;

		throughput = F * Microfacet::WeightBeckmannWalter(alpha, abs(wiDotM), abs(wo.z), abs(wi.z), m.z);
		pdf = Microfacet::SampleWalterReflectionPdf(alpha, m.z, fabs(wiDotM));

		return Sampler::IsPdfValid(pdf);
	}
};