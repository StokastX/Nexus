#pragma once

#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"

// Quaternion class
// See https://en.wikipedia.org/wiki/Quaternion
struct Quat4
{
    Quat4() : data(make_float4(0.0f, 0.0f, 0.0f, 1.0f)) {}
    __host__ __device__ Quat4(const float3& v, float w): data(make_float4(v.x, v.y, v.z, w)) {}
    __host__ __device__ Quat4(const float4& v) : data(v) {}
    __host__ __device__ Quat4(float x, float y, float z, float w) : data(make_float4(x, y, z, w)) {}

    __host__ __device__ Quat4 Conjugate()
    {
        return Quat4(-data.x, -data.y, -data.z, data.w);
    }

    __host__ __device__ float Real()
    {
        return data.w;
    }

    __host__ __device__ float3 Complex()
    {
        return make_float3(data.x, data.y, data.z);
    }

    __host__ __device__ float Norm()
    {
        return sqrtf(dot(data, data));
    }

    __host__ __device__ Quat4 Normalize()
    {
        return Quat4(data / Norm());
    }

    __host__ __device__ Quat4 Inverse()
    {
        return Conjugate() / Norm();
    }

    // See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternions_as_rotations
    // Rotate unit vector v. The quaternion must be normalized.
    __host__ __device__ float3 Rotate(float3 v)
    {
        float3 u = Complex();
        float3 t = 2.0f * cross(u, v);
        return v + data.w * t + cross(u, t);
    }

    // Rotation of angle a around unit vector v
    static __host__ __device__ Quat4 AngleAxis(float a, float3 v)
    {
        float s = sinf(a * 0.5f);
        return Quat4(v * s, cosf(a * 0.5f));
    }

    // Rotation from unit vector v to the X-axis
    static __host__ __device__ Quat4 RotationToXAxis(float3 v)
    {
        if (v.x < -0.99999f)
            return Quat4(0.0f, 1.0f, 0.0f, 0.0f);

        return Quat4(0.0f, v.z, -v.y, 1.0f + v.x);
    }

    // Rotation from unit vector v to the Y-axis
    static __host__ __device__ Quat4 RotationToYAxis(float3 v)
    {
        if (v.y < -0.99999f)
            return Quat4(0.0f, 0.0f, 1.0f, 0.0f);

        return Quat4(-v.z, 0.0f, v.x, 1.0f + v.y);
    }

    // Rotation from unit vector v to the Z-axis
    static __host__ __device__ Quat4 RotationToZAxis(float3 v)
    {
        if (v.z < -0.99999f)
            return Quat4(1.0f, 0.0f, 0.0f, 0.0f);

        return Quat4(v.y, -v.x, 0.0f, 1.0f + v.z);
    }

    // From https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    // Rotation from unit vector v1 to unit vector v2
    static __host__ __device__ Quat4 RotationBetween(float3 v1, float3 v2)
    {
        float3 a = cross(v1, v2);

        return Quat4(a.x, a.y, a.z, 1.0f + dot(v1, v2));
    }

    // See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    __host__ __device__ Quat4 operator*(const Quat4& other)
    {
        return Quat4(
            data.w * other.data.x + data.x * other.data.w + data.y * other.data.z - data.z * other.data.y,
            data.w * other.data.y - data.x * other.data.z + data.y * other.data.w + data.z * other.data.x,
            data.w * other.data.z + data.x * other.data.y - data.y * other.data.x + data.z * other.data.w,
            data.w * other.data.w - data.x * other.data.x - data.y * other.data.y - data.z * other.data.z
        );
    }

    __host__ __device__ Quat4 operator*(float s)
    {
        return Quat4(data * s);
    }

    __host__ __device__ Quat4 operator+(const Quat4& other)
    {
        return Quat4(data + other.data);
    }

    __host__ __device__ Quat4 operator-(const Quat4& other)
    {
        return Quat4(data - other.data);
    }

    __host__ __device__ Quat4 operator/(float s)
    {
        return Quat4(data / s);
    }

    float4 data;
};