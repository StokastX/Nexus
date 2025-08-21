#pragma once

#include "Cuda/Random.cuh"
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Geometry/Ray.cuh"
#include "Utils/cuda_math.h"
#include "Microfacet.cuh"
#include "Fresnel.cuh"

struct D_ConductorBSDF
{
	float2 alpha;

	inline __device__ void PrepareBSDFData(const float3& wi, const D_Material& material)
	{
		alpha.x = Square(material.roughness) * sqrtf(2.0f / (1.0f + Square(1.0f - material.anisotropy)));
		alpha.y = (1.0f - material.anisotropy) * alpha.x;
		alpha = clamp(alpha, 1.0e-4f, 1.0f);
	}

	inline __device__ bool Eval(const D_Material& material, const float3& wi, const float3& wo, float3& throughput, float& pdf)
	{
		const float3 m = normalize(wo + wi);
		const float wiDotM = dot(wi, m);
		float3 F82 = material.specularColor * Fresnel::SchlickMetallicReflectance(material.baseColor, fabs(wiDotM));
		float3 F = material.specularWeight * Fresnel::SchlickMetallicF82(material.baseColor, F82, fabs(wiDotM));
		const float D = Microfacet::D_GGXAnisotropic(m, alpha);
		const float G1 = Microfacet::G1_GGX(wi, alpha);
		const float G2 = Microfacet::G2_GGX(wi, wo, alpha);

		// BRDF times woDotN
		throughput = F * G2 * D / (4.0f * fabs(wi.z));

		pdf = Microfacet::ReflectionPdf_GGX(D, G1, fabs(wi.z));

		return Sampler::IsPdfValid(pdf);
	}

	inline __device__ bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{
		const float3 m = Microfacet::SampleVNDF_GGX(wi, alpha, rngState);

		const float wiDotM = dot(wi, m);

		float3 F82 = material.specularColor * Fresnel::SchlickMetallicReflectance(material.baseColor, fabs(wiDotM));
		float3 F = material.specularWeight * Fresnel::SchlickMetallicF82(material.baseColor, F82, fabs(wiDotM));

		wo = reflect(-wi, m);

		// If the new ray is under the hemisphere, return
		if (wo.z * wi.z < 0.0f)
			return false;

		const float D = Microfacet::D_GGXAnisotropic(m, alpha);
		const float G1 = Microfacet::G1_GGX(wi, alpha);
		const float G2 = Microfacet::G2_GGX(wi, wo, alpha);

		throughput = F * G2 / G1;
		pdf = Microfacet::ReflectionPdf_GGX(D, G1, fabs(wi.z));

		return Sampler::IsPdfValid(pdf);
	}
};