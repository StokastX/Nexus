#pragma once

#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Geometry/Ray.cuh"
#include "Cuda/Random.cuh"
#include "Microfacet.cuh"
#include "Fresnel.cuh"
#include "Cuda/Sampler.cuh"

/* 
 * Rough plastic BSDF. It uses a basic non physically accurate microfacet model (specular + diffuse)
 * It would be great to implement a layered model for this material, as described in
 * "Arbitrarily Layered Micro-Facet Surfaces" by Weidlich and Wilkie
 * See https://www.cg.tuwien.ac.at/research/publications/2007/weidlich_2007_almfs/weidlich_2007_almfs-paper.pdf
 */
struct D_PlasticBSDF
{
	float eta;
	float2 alpha;

	inline __device__ void PrepareBSDFData(const float3& wi,  const D_Material& material)
	{
		eta = wi.z < 0.0f ? material.ior : 1 / material.ior;
		alpha.x = Square(material.roughness) * sqrtf(2.0f / (1.0f + Square(1.0f - material.anisotropy)));
		alpha.y = (1.0f - material.anisotropy) * alpha.x;
		alpha = clamp(alpha, 1.0e-4f, 1.0f);
	}

	// Evaluation function for a shadow ray
	inline __device__ bool Eval(const D_Material& material, const float3& wi, const float3& wo, float3& bsdf, float& pdf)
	{
		const float3 m = normalize(wo + wi);
		const float wiDotM = dot(wi, m);
		const float F = Fresnel::DielectricReflectance(eta, wiDotM, material.specularWeight);
		const float3 FTinted = material.specularColor * F;
		const float G1 = Microfacet::G1_GGX(wi, alpha);
		const float G2 = Microfacet::G2_GGX(wi, wo, alpha);
		const float D = Microfacet::D_GGXAnisotropic(m, alpha);

		// BRDF times woDotN
		const float3 brdf = FTinted * G2 * D / (4.0f * fabs(wi.z));

		// Diffuse bounce
		const float3 btdf = (1.0f - F) * material.baseColor * INV_PI * fabs(wo.z);

		bsdf = brdf + btdf;

		const float pdfSpecular = Microfacet::ReflectionPdf_GGX(D, G1, fabs(wi.z));

		// cos(theta) / PI
		const float pdfDiffuse = fabs(wo.z) * INV_PI;

		pdf = F * pdfSpecular + (1.0f - F) * pdfDiffuse;
		
		return Sampler::IsPdfValid(pdf);
	}

	inline __device__ bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{
		const float3 m = Microfacet::SampleVNDF_GGX(wi, alpha, rngState);

		const float wiDotM = dot(wi, m);

		float F = Fresnel::DielectricReflectance(eta, wiDotM, material.specularWeight);

		// Randomly select a specular or diffuse ray based on Fresnel reflectance
		if (Random::Rand(rngState) < F)
		{
			// Specular
			wo = reflect(-wi, m);

			// If the new ray is under the hemisphere, return
			if (wo.z * wi.z < 0.0f)
				return false;

			const float3 FTinted = material.specularColor * F;
			const float D = Microfacet::D_GGXAnisotropic(m, alpha);
			const float G1 = Microfacet::G1_GGX(wi, alpha);
			const float G2 = Microfacet::G2_GGX(wi, wo, alpha);

			throughput = G2 * FTinted / (G1 * F);
			pdf = F * Microfacet::ReflectionPdf_GGX(D, G1, fabs(wi.z));
		}

		else
		{
			//Diffuse
			wo = Random::RandomCosineHemisphere(rngState);
			throughput = material.baseColor;
			pdf = (1.0f - F) * INV_PI * wo.z;
		}

		return Sampler::IsPdfValid(pdf);
	}
};
