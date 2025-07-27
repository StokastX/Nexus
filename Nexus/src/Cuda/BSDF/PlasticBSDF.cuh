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
	float alpha;

	inline __device__ void PrepareBSDFData(const float3& wi,  const D_Material& material)
	{
		alpha = clamp((1.2f - 0.2f * sqrtf(fabs(wi.z))) * material.plastic.roughness * material.plastic.roughness, 1.0e-4f, 1.0f);
		eta = wi.z < 0.0f ? material.plastic.ior : 1 / material.plastic.ior;
	}

	// Evaluation function for a shadow ray
	inline __device__ bool Eval(const D_Material& material, const float3& wi, const float3& wo, float3& throughput, float& pdf)
	{
		const float wiDotN = wi.z;
		const float woDotN = wo.z;

		//const bool reflected = wiDotN * woDotN > 0.0f;

		//if (!reflected)
		//	return false;

		const float3 m = normalize(wo + wi);
		float cosThetaT;
		const float wiDotM = dot(wi, m);
		const float F = Fresnel::DieletricReflectance(1.0f / material.plastic.ior, wiDotM, cosThetaT);
		const float G = Microfacet::Smith_G2(alpha, fabs(woDotN), fabs(wiDotN));
		const float D = Microfacet::BeckmannD(alpha, m.z);

		// BRDF times woDotN
		const float3 brdf = make_float3(F * G * D / (4.0f * fabs(wiDotN)));

		// Diffuse bounce
		const float3 btdf = (1.0f - F) * material.plastic.albedo * INV_PI * fabs(wo.z);

		throughput = brdf + btdf;

		// pm * jacobian = pm * || dWhr / dWo ||
		// We can replace woDotM with wiDotM since for a reflection both are equal
		const float pdfSpecular = D * fabs(m.z) / (4.0f * fabs(wiDotM));

		// cos(theta) / PI
		const float pdfDiffuse = fabs(wo.z) * INV_PI;

		pdf = F * pdfSpecular + (1.0f - F) * pdfDiffuse;
		
		return Sampler::IsPdfValid(pdf);
	}

	inline __device__ bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{
		const float3 m = Microfacet::SampleSpecularHalfBeckWalt(alpha, rngState);

		const float wiDotM = dot(wi, m);

		float cosThetaT;
		const float fr = Fresnel::DieletricReflectance(1.0f / material.plastic.ior, wiDotM, cosThetaT);

		// Randomly select a specular or diffuse ray based on Fresnel reflectance
		if (Random::Rand(rngState) < fr)
		{
			// Specular
			wo = reflect(-wi, m);

			// If the new ray is under the hemisphere, return
			if (wo.z * wi.z < 0.0f)
				return false;

			const float weight = Microfacet::WeightBeckmannWalter(alpha, fabs(wiDotM), fabs(wo.z), fabs(wi.z), m.z);

			// We dont need to include the Fresnel term since it's already included when
			// we select between reflection and transmission (see paper page 7)
			throughput = make_float3(weight); // * F / fr
			pdf = fr * Microfacet::SampleWalterReflectionPdf(alpha, m.z, fabs(wiDotM));
		}

		else
		{
			//Diffuse
			wo = Random::RandomCosineHemisphere(rngState);
			throughput = material.plastic.albedo;
			// Same here, we don't need to include the Fresnel term
			//throughput = throughput * (1.0f - F) / (1.0f - fr)
			pdf = (1.0f - fr) * INV_PI * wo.z;
		}

		return Sampler::IsPdfValid(pdf);
	}
};
