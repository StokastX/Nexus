#pragma once
#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"

class Fresnel
{
public:
	// Fresnel reflectance for dieletric materials. See https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
	inline static __device__ float DieletricReflectance(float eta, float cosThetaI, float& cosThetaT)
	{
		if (cosThetaI < 0.0f)
		{
			eta = 1.0f / eta;
			cosThetaI = -cosThetaI;
		}

		const float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);

		if (sinThetaTSq > 1.0f)
		{
			cosThetaT = 0.0f;
			return 1.0f;
		}

		cosThetaT = sqrt(fmax(0.0f, 1.0f - sinThetaTSq));

		const float Rparl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
		const float Rperp = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaT);

		return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
	}

	// Schlick reflectance approximation
	inline static __device__ float SchlickDielectricReflectance(const float F0, const float F90, const float cosThetaI)
	{
		return F0 + (F90 - F0) * pow(1.0f - cosThetaI, 5.0f);
	}

	// Modified fresnel reflectance at grazing angles (90�) depending on R0. See https://boksajak.github.io/files/CrashCourseBRDF.pdf
	inline static __device__ float ShadowedF90(const float3& F0) {
		const float t = (1.0f / 0.04f);
		return min(1.0f, t * Luminance(F0));
	}

	inline static __device__ float ComplexReflectance(float cosThetaI, const float eta, const float k)
	{
		cosThetaI = clamp(cosThetaI, 0.0f, 1.0f);

		float cosThetaISq = cosThetaI * cosThetaI;
		float sinThetaISq = max(1.0f - cosThetaISq, 0.0f);
		float sinThetaIQu = sinThetaISq * sinThetaISq;

		float innerTerm = eta * eta - k * k - sinThetaISq;
		float aSqPlusBSq = sqrtf(max(innerTerm * innerTerm + 4.0f * eta * eta * k * k, 0.0f));
		float a = sqrtf(max((aSqPlusBSq + innerTerm) * 0.5f, 0.0f));

		float Rs = ((aSqPlusBSq + cosThetaISq) - (2.0f * a * cosThetaI)) /
			((aSqPlusBSq + cosThetaISq) + (2.0f * a * cosThetaI));
		float Rp = ((cosThetaISq * aSqPlusBSq + sinThetaIQu) - (2.0f * a * cosThetaI * sinThetaISq)) /
			((cosThetaISq * aSqPlusBSq + sinThetaIQu) + (2.0f * a * cosThetaI * sinThetaISq));

		return 0.5f * (Rs + Rs * Rp);
	}

	inline static __device__ float3 ComplexReflectance(float cosThetaI, const float3 eta, const float3 k)
	{
		return make_float3(ComplexReflectance(cosThetaI, eta.x, k.x), ComplexReflectance(cosThetaI, eta.y, k.y), ComplexReflectance(cosThetaI, eta.z, k.z));
	}

private:
	inline static __device__ float Luminance(const float3& rgb)
	{
		return dot(rgb, make_float3(0.2126f, 0.7152f, 0.0722f));
	}

};