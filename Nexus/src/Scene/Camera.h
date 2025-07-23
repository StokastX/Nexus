#pragma once

#include <vector>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "Geometry/Ray.h"
#include "Cuda/Scene/Camera.cuh"

class Camera 
{
public:
	Camera(float verticalFOV, uint32_t width, uint32_t height);
	Camera(float3 position, float3 forward, float verticalFOV, uint32_t width, uint32_t height, float focusDistance, float defocusAngle);

	void OnUpdate(float ts);
	void OnResize(uint32_t width, uint32_t height);

	void SetHorizontalFOV(float horizontalFOV);

	float GetRotationSpeed();
	float& GetHorizontalFOV() { return m_HorizontalFOV; }
	float& GetDefocusAngle() { return m_DefocusAngle; }
	float& GetFocusDist() { return m_FocusDist; }
	uint32_t GetViewportWidth() { return m_ViewportWidth; }
	uint32_t GetViewportHeight() { return m_ViewportHeight; }
	float3& GetPosition() { return m_Position; }
	float3& GetForwardDirection() { return m_ForwardDirection; }
	float3& GetRightDirection() { return m_RightDirection; }
	Ray RayThroughPixel(int2 pixel);

	bool IsInvalid() { return m_Invalid; }
	void SetInvalid(bool invalid) { m_Invalid = invalid; }
	void Invalidate() { m_Invalid = true; }

	static D_Camera ToDevice(const Camera& camera);

private:
	float2 m_LastMousePosition{ 0.0f, 0.0 };

	float m_HorizontalFOV;
	float m_DefocusAngle;
	float m_FocusDist;
	uint32_t m_ViewportWidth;
	uint32_t m_ViewportHeight;
	float3 m_Position;
	float3 m_ForwardDirection;
	float3 m_RightDirection;

	bool m_Invalid = true;
};
