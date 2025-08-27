#pragma once

#include <vector>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "Geometry/Ray.h"
#include "Cuda/Scene/Camera.cuh"

class Camera 
{
public:
	Camera(float verticalFOV, uint2 resolution);
	Camera(float3 position, float3 forward, float verticalFOV, uint2 resolution, float focusDistance, float defocusAngle);

	void OnUpdate(float ts);
	void OnResize(uint2 resolution);

	float GetRotationSpeed();
	float& GetHorizontalFOV() { return m_HorizontalFOV; }
	void SetHorizontalFOV(float horizontalFOV) { m_HorizontalFOV = horizontalFOV; }
	float& GetDefocusAngle() { return m_DefocusAngle; }
	void SetDefocusAngle(float defocusAngle) { m_DefocusAngle = defocusAngle; }
	float& GetFocusDist() { return m_FocusDist; }
	void SetFocusDist(float focusDist) { m_FocusDist = focusDist; }
	uint2 GetResolution() { return m_Resolution; }
	float3& GetPosition() { return m_Position; }
	void SetPosition(const float3& position) { m_Position = position; }
	float3& GetForwardDirection() { return m_ForwardDirection; }
	void SetForwardDirection(const float3& direction) { m_ForwardDirection = direction; }
	float3& GetRightDirection() { return m_RightDirection; }
	void SetRightDirection(const float3& rightDirection) { m_RightDirection = rightDirection; }
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
	uint2 m_Resolution;
	float3 m_Position;
	float3 m_ForwardDirection;
	float3 m_RightDirection;

	bool m_Invalid = true;
};
