#pragma once

#include <cuda_runtime_api.h>

namespace ColorUtils
{
    constexpr float gamma = 2.2f;

	enum struct ToneMapping {
        NONE,
        ACES,
        UNCHARTED2,
        AGX_DEFAULT,
        AGX_GOLDEN,
        AGX_PUNCHY
	};

	static const char *ToneMappingNames[] = {
		"None",
		"ACES",
		"Uncharted 2",
        "AGX (default)",
        "AGX (golden)",
        "AGX (punchy)"
	};

	inline __host__ __device__ float3 LinearToGamma(const float3& color)
	{
        constexpr float invGamma = 1.0f / gamma;
		return make_float3(powf(color.x, invGamma), powf(color.y, invGamma), powf(color.z, invGamma));
	}

	inline __host__ __device__ float3 GammaToLinear(const float3& color)
	{
		return make_float3(powf(color.x, gamma), powf(color.y, gamma), powf(color.z, gamma));
	}

	inline __host__ __device__ float Luminance(const float3& rgb)
	{
		return dot(rgb, make_float3(0.2126f, 0.7152f, 0.0722f));
	}

    inline __host__ __device__ uint32_t ToColorUInt(float3 color)
    {
        float4 clamped = clamp(make_float4(color, 1.0f), 0.0f, 1.0f);
        uint8_t red = (uint8_t)(clamped.x * 255.0f);
        uint8_t green = (uint8_t)(clamped.y * 255.0f);
        uint8_t blue = (uint8_t)(clamped.z * 255.0f);
        uint8_t alpha = (uint8_t)(clamped.w * 255.0f);

        return alpha << 24 | blue << 16 | green << 8 | red;
    }

    // Approximated ACES tonemapping by Krzysztof Narkowicz. See https://graphics-programming.org/resources/tonemapping/index.html
    inline __host__ __device__ float3 ACESToneMapping(float3 color)
    {
        constexpr float a = 2.51f;
        constexpr float b = 0.03f;
        constexpr float c = 2.43f;
        constexpr float d = 0.59f;
        constexpr float e = 0.14f;
        return (color * (a * color + b)) / (color * (c * color + d) + e);
    }

    template<typename T>
    static inline __host__ __device__ T Uncharted2ToneMappingPartial(T x)
    {
        constexpr float a = 0.15f;
        constexpr float b = 0.50f;
        constexpr float c = 0.10f;
        constexpr float d = 0.20f;
        constexpr float e = 0.02f;
        constexpr float f = 0.30f;
        return (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f;
    }

    inline __host__ __device__ float3 Uncharted2ToneMapping(float3 color)
    {
        constexpr float white = 11.2f;
        return Uncharted2ToneMappingPartial(1.6f * color) / Uncharted2ToneMappingPartial(white);
    }

    // Minimal AgX implementation by Benjamin Wrensch
    // See https://iolite-engine.com/blog_posts/minimal_agx_implementation

    // 7th order polynomial approximation. Mean error^2: 1.85907662e-06
    inline __host__ __device__ float3 AgxDefaultContrastApprox(float3 x)
    {
        float3 x2 = x * x;
        float3 x4 = x2 * x2;
        float3 x6 = x4 * x2;

        return -17.86 * x6 * x + 78.01 * x6 - 126.7 * x4 * x + 92.06 * x4 - 28.72 * x2 * x + 4.361 * x2 - 0.1718 * x + 0.002857;
    }

    inline __host__ __device__ float3 Agx(float3 val)
    {
        constexpr float3 agxMatRow1 = {0.842479062253094, 0.0423282422610123, 0.0423756549057051};
        constexpr float3 agxMatRow2 = {0.0784335999999992, 0.878468636469772, 0.0784336};
        constexpr float3 agxMatRow3 = {0.0792237451477643, 0.0791661274605434, 0.879142973793104};

        constexpr float minEv = -12.47393f;
        constexpr float maxEv = 4.026069f;

        // Input transform (inset)
        val = make_float3(dot(agxMatRow1, val), dot(agxMatRow2, val), dot(agxMatRow3, val));

        // Log2 space encoding
        val = clamp(make_float3(log2f(val.x), log2f(val.y), log2f(val.z)), minEv, maxEv);
        val = (val - minEv) / (maxEv - minEv);

        // Apply sigmoid function approximation
        val = AgxDefaultContrastApprox(val);

        return val;
    }

    inline __host__ __device__ float3 AgxEotf(float3 val)
    {
        constexpr float3 agxMatInvRow1 = {1.19687900512017, -0.0528968517574562, -0.0529716355144438};
        constexpr float3 agxMatInvRow2 = {-0.0980208811401368, 1.15190312990417, -0.0980434501171241};
        constexpr float3 agxMatInvRow3 = {-0.0990297440797205, -0.0989611768448433, 1.15107367264116};

        // Inverse input transform (outset)
        val = make_float3(dot(agxMatInvRow1, val), dot(agxMatInvRow2, val), dot(agxMatInvRow3, val));

        // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
        // NOTE: We're linearizing the output here. Comment/adjust when
        // *not* using a sRGB render target
        val = GammaToLinear(val);

        return val;
    }

    inline __host__ __device__ float3 AgxLook(float3 val, ToneMapping toneMapping)
    {
        float3 offset = make_float3(0.0);
        float3 slope, power;
        float sat;

        switch (toneMapping)
        {
        case ToneMapping::AGX_GOLDEN:
            slope = make_float3(1.0, 0.9, 0.5);
            power = make_float3(0.8);
            sat = 0.8;
            break;
        case ToneMapping::AGX_PUNCHY:
            slope = make_float3(1.0);
            power = make_float3(1.35, 1.35, 1.35);
            sat = 1.4;
            break;
        default:
            slope = make_float3(1.0);
            power = make_float3(1.0);
            sat = 1.0;
            break;
        }

        // ASC CDL
        float3 a = val * slope + offset;
        val = make_float3(powf(a.x, power.x), powf(a.y, power.y), powf(a.z, power.z));

        float luma = Luminance(val);

        return luma + sat * (val - luma);
    }

    inline __host__ __device__ float3 AgxToneMapping(float3 color, ToneMapping toneMapping)
    {
        color = Agx(color);
        color = AgxLook(color, toneMapping);
        return AgxEotf(color);
    }

    inline __host__ __device__ float3 HeatmapColor(int boundsHit, int maxHit = 60)
    {
        float t = fminf((float)boundsHit / maxHit, 1.0f);

        if (t < 0.5f)
        {
            // Blue to Green
            float localT = t * 2.0f; // remap to [0,1]
            return make_float3(0.0f, localT, 1.0f - localT);
        }
        else
        {
            // Green to Red
            float localT = (t - 0.5f) * 2.0f; // remap to [0,1]
            return make_float3(localT, 1.0f - localT, 0.0f);
        }
    }
}