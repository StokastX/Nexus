#pragma once
#include <iostream>
#include "Utils/cuda_math.h"
#include "Math/Mat4.h"

// Adapted from https://learnopengl.com
class Shader
{
public:
	Shader(const std::string& vertexPath, const std::string& fragmentPath);
	void Use();
	void SetBool(const std::string& name, bool value);
	void SetInt(const std::string& name, int value);
	void SetFloat(const std::string& name, float value);
	void SetVec2(const std::string& name, const float2& value);
	void SetVec3(const std::string& name, const float3& value);
	void SetVec4(const std::string& name, const float4& value);
	void SetMat4(const std::string& name, const Mat4& mat);
	uint32_t GetId() { return m_Id; }

private:
	void CheckCompileErrors(uint32_t shader, const std::string& type);

private:
	uint32_t m_Id;
};