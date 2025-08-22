#pragma once

#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"
#include "Cuda/Utils.cuh"
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Geometry/Ray.cuh"
#include "Cuda/Random.cuh"
#include "Microfacet.cuh"
#include "Fresnel.cuh"
#include "Cuda/Sampler.cuh"

/* 
 *	Rough dielectric BSDF based on the paper "Microfacet Models for Refraction through Rough Surfaces"
 *	See https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
 */
struct D_DielectricBSDF
{
	float eta;
	float2 alpha;

	inline __device__ void PrepareBSDFData(const float3& wi,  const D_Material& material)
	{
		eta = wi.z < 0.0f ? material.ior : 1.0f / material.ior;

		// OpenPBR alpha mapping
		alpha.x = Square(material.roughness) * sqrtf(2.0f / (1.0f + Square(1.0f - material.anisotropy)));
		alpha.y = (1.0f - material.anisotropy) * alpha.x;
		alpha = clamp(alpha, 1.0e-4f, 1.0f);
	}

	inline __device__ bool Eval(const D_Material& material, const float3& wi, const float3& wo, float3& throughput, float& pdf)
	{
		const bool reflected = wi.z * wo.z > 0.0f;

		float3 m;
		if (reflected)
			m = normalize(wo + wi);
		else
			m = normalize(wi * eta + wo); // We don't care about the sign of m

		float cosThetaT;
		const float wiDotM = dot(wi, m);
		const float woDotM = dot(wo, m);

		const float F = Fresnel::DielectricReflectance(eta, fabs(wiDotM), material.specularWeight);
		const float G1 = Microfacet::G1_GGX(wi, alpha);
		const float G2 = Microfacet::G2_GGX(wi, wo, alpha);
		const float D = Microfacet::D_GGXAnisotropic(m, alpha);

		if (reflected)
		{
			// BSDF times woDotN
			throughput = make_float3(F * G2 * D / (4.0f * fabs(wi.z)));
			pdf = F * Microfacet::ReflectionPdf_GGX(D, G1, fabs(wi.z));
		}
		else
		{
			// wiDotM and woDotM always have an opposite sign, this is why we don't care about the sign of m
			throughput = (1.0f - F) * G2 * D * fabs(wiDotM * woDotM) / (fabs(wi.z) * Square(eta * wiDotM + woDotM)) * material.baseColor;
			pdf = (1.0f - F) * Microfacet::RefractionPdf_GGX(D, G1, eta, wi.z, wiDotM, woDotM);
		}

		return Sampler::IsPdfValid(pdf);
	}

	inline __device__ bool Sample(const D_Material& material, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{
		// Very ugly change of sign because we can only sample in the upper hemisphere
		// Requiring wi.z to be always positive would fix this, but I would need to change a few things in the shading kernel
		// Also, VNDF sampling allows to sample below the horizon when wi.z is negative, so I'm not sure how this is compatible with refraction
		float3 w = wi;
		if (wi.z < 0)
			w.z = -w.z;
		float3 m = Microfacet::SampleVNDF_GGX(w, alpha, rngState);
		if (wi.z < 0)
			m.z = -m.z;

		// Should always be positive since we sample visible normals
		const float wiDotM = dot(wi, m);

		const float F = Fresnel::DielectricReflectance(eta, wiDotM, material.specularWeight);

		// Randomly select a reflected or transmitted ray based on Fresnel reflectance
		const bool reflection = Random::Rand(rngState) < F;

		if (reflection)
		{
			wo = reflect(-wi, m);

			// If the new ray is under the hemisphere, return
			if (wo.z * wi.z < 0.0f)
				return false;
		}

		else
		{
			// Refraction
			float cosThetaT = sqrtf(1.0f - Square(eta) * (1.0f - Square(wiDotM)));
			wo = (eta * wiDotM - Utils::SgnE(wiDotM) * cosThetaT) * m - eta * wi;

			if (wo.z * wi.z > 0.0f)
				return false;
		}

		const float D = Microfacet::D_GGXAnisotropic(m, alpha);
		const float G1 = Microfacet::G1_GGX(wi, alpha);
		const float G2 = Microfacet::G2_GGX(wi, wo, alpha);

		if (reflection)
		{
			throughput = make_float3(G2 / G1);
			pdf = F * Microfacet::ReflectionPdf_GGX(D, G1, fabs(wi.z));
		}
		else
		{
			const float woDotM = dot(wo, m);
			throughput = material.baseColor * make_float3(G2 / (G1));
			pdf = (1.0f - F) * Microfacet::RefractionPdf_GGX(D, G1, eta, wi.z, wiDotM, woDotM);
		}
		return Sampler::IsPdfValid(pdf);
	}
};