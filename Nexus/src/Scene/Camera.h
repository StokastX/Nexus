#pragma once

#include <vector>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "Geometry/Ray.h"
#include "Cuda/Scene/Camera.cuh"
#include "Math/Mat4.h"
#include "Device/DeviceTraits.h"

namespace Nexus {

	class Camera;

	// Declared up front so the class below can befriend it: the conversion reads private
	// state, which a non-intrusive trait otherwise cannot reach.
	template<>
	struct DeviceTraits<Camera>;

	class Camera
	{
	public:
		Camera(uint2 resolution);
		Camera(float3 position, float3 forward, float verticalFOV, uint2 resolution, float focusDistance, float defocusAngle);

		void OnUpdate(float ts);
		void OnResize(uint2 resolution);

		float GetRotationSpeed();
		float& GetHorizontalFOV() { return m_HorizontalFOV; }
		void SetHorizontalFOV(float horizontalFOV) { m_HorizontalFOV = horizontalFOV; }
		float GetVerticalFOV();
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
		Mat4 GetTransform();
		Mat4 GetViewMatrix();
		Mat4 GetProjectionMatrix();
		void SetRightDirection(const float3& rightDirection) { m_RightDirection = rightDirection; }
		Ray RayThroughPixel(int2 pixel);

		/*
		 * Whether the view changed since the last frame was accumulated.
		 *
		 * Not an upload signal. The camera crosses to the device inside D_Scene, which ToDevice
		 * rebuilds every frame, so there is never anything to flush -- the only consequence of a
		 * camera change is that the accumulated image has to restart.
		 */
		bool Changed() const { return m_Changed; }
		void MarkChanged() { m_Changed = true; }
		void ClearChanged() { m_Changed = false; }

		friend struct DeviceTraits<Camera>;

	private:
		float2 m_LastMousePosition = make_float2(0.0f, 0.0);
		float m_HorizontalFOV = 45.0f;
		float m_DefocusAngle = 0.0f;
		float m_FocusDist = 1.0f;
		uint2 m_Resolution;
		float3 m_Position = make_float3(0.0f, 4.0f, 14.0f);
		float3 m_ForwardDirection = make_float3(0.0f, 0.0f, -1.0f);
		float3 m_RightDirection = cross(m_ForwardDirection, make_float3(0.0f, 1.0f, 0.0f));

		bool m_Changed = true;
	};


	template<>
	struct DeviceTraits<Camera>
	{
		using DeviceType = D_Camera;

		static D_Camera ToDevice(const Camera& camera);
	};

}
