#include "Camera.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <math.h>

#include "Utils/Utils.h"
#include "Utils/cuda_math.h"
#include "Math/Quat4.h"
#include "Input.h"
#include "Cuda/PathTracer/PathTracer.cuh"


Camera::Camera(float horizontalFOV, uint2 resolution)
{
	m_HorizontalFOV = horizontalFOV;
	m_DefocusAngle = 10.0f;
	m_FocusDist = 5.0f;
	m_Resolution = resolution;
	m_Position = make_float3(0.0f, 0.0f, 2.0f);
	m_ForwardDirection = make_float3(0.0f, 0.0f, -1.0f);
	m_RightDirection = make_float3(1.0f, 0.0f, 0.0f);
}

Camera::Camera(float3 position, float3 forward, float horizontalFOV, uint2 resolution, float focusDistance, float defocusAngle)
	:m_Position(position), m_ForwardDirection(forward), m_HorizontalFOV(horizontalFOV), m_Resolution(resolution),
	m_RightDirection(cross(m_ForwardDirection, make_float3(0.0f, 1.0f, 0.0f))), m_FocusDist(focusDistance), m_DefocusAngle(defocusAngle)
{

}


void Camera::OnUpdate(float ts)
{
	float2 mousePos = Input::GetMousePosition();
	float2 delta = (mousePos - m_LastMousePosition) * 2.0f;
	m_LastMousePosition = mousePos;

	if (!Input::IsMouseButtonDown(GLFW_MOUSE_BUTTON_RIGHT))
	{
		Input::SetCursorMode(GLFW_CURSOR_NORMAL);
		return;
	}

	Input::SetCursorMode(GLFW_CURSOR_DISABLED);

	float3 upDirection = make_float3(0.0f, 1.0f, 0.0f);

	float speed = 0.003f;

	if (Input::IsKeyDown(GLFW_KEY_W))
	{
		m_Position += ts * speed * m_ForwardDirection;
		m_Invalid = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_S))
	{
		m_Position -= ts * speed * m_ForwardDirection;
		m_Invalid = true;
	}
	if (Input::IsKeyDown(GLFW_KEY_A))
	{
		m_Position -= ts * speed * m_RightDirection;
		m_Invalid = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_D))
	{
		m_Position += ts * speed * m_RightDirection;
		m_Invalid = true;
	}
	if (Input::IsKeyDown(GLFW_KEY_Q))
	{
		m_Position -= ts * speed * upDirection;
		m_Invalid = true;
	}
	else if (Input::IsKeyDown(GLFW_KEY_E))
	{
		m_Position += ts * speed * upDirection;
		m_Invalid = true;
	}

	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * GetRotationSpeed();
		float yawDelta = delta.x * GetRotationSpeed();

		Quat4 q = (Quat4::AngleAxis(-pitchDelta, m_RightDirection) * Quat4::AngleAxis(-yawDelta, make_float3(0.0f, 1.0f, 0.0f))).Normalized();
		m_ForwardDirection = q.Rotate(m_ForwardDirection);
		m_RightDirection = normalize(cross(m_ForwardDirection, upDirection));

		m_Invalid = true;
	}
}

void Camera::OnResize(uint2 resolution)
{
	if (resolution.x == m_Resolution.x && resolution.y == m_Resolution.y)
		return;

	m_Resolution.x = resolution.x;
	m_Resolution.y = resolution.y;
	Invalidate();
}

float Camera::GetRotationSpeed()
{
	return 0.0008f;
}

// Returns the ray traversing the given pixel
Ray Camera::RayThroughPixel(int2 pixel)
{
	float3 upDirection = cross(m_RightDirection, m_ForwardDirection);

	float x = pixel.x / (float)m_Resolution.x;
	float y = pixel.y / (float)m_Resolution.y;

	float aspectRatio = m_Resolution.x / (float)m_Resolution.y;
	float halfHeight = m_FocusDist * tanf(m_HorizontalFOV / 2.0f * PI / 180.0f);
	float halfWidth = aspectRatio * halfHeight;

	float3 viewportX = 2 * halfWidth * m_RightDirection;
	float3 viewportY = 2 * halfHeight * upDirection;
	float3 lowerLeftCorner = m_Position - viewportX / 2.0f - viewportY / 2.0f + m_ForwardDirection * m_FocusDist;
	float3 direction = normalize(lowerLeftCorner + x * viewportX + y * viewportY - m_Position);

	return Ray(m_Position, direction);
}

D_Camera Camera::ToDevice(const Camera& camera)
{
	D_Camera deviceCamera;

	float3 forwardDirection = camera.m_ForwardDirection;
	float3 upDirection = cross(camera.m_RightDirection, forwardDirection);
	float aspectRatio = camera.m_Resolution.x / (float)camera.m_Resolution.y;
	float halfWidth = camera.m_FocusDist * tanf(camera.m_HorizontalFOV / 2.0f * PI / 180.0f);
	float halfHeight = halfWidth / aspectRatio;

	float3 viewportX = 2 * halfWidth * camera.m_RightDirection;
	float3 viewportY = 2 * halfHeight * upDirection;
	float3 lowerLeftCorner = camera.m_Position - viewportX / 2.0f - viewportY / 2.0f + forwardDirection * camera.m_FocusDist;

	float lensRadius = camera.m_FocusDist * tanf(camera.m_DefocusAngle / 2.0f * PI / 180.0f);

	deviceCamera.position = camera.m_Position;
	deviceCamera.right = camera.m_RightDirection;
	deviceCamera.up = upDirection;
	deviceCamera.lensRadius = lensRadius;
	deviceCamera.lowerLeftCorner = lowerLeftCorner;
	deviceCamera.viewportX = viewportX;
	deviceCamera.viewportY = viewportY;
	deviceCamera.resolution = camera.m_Resolution;

	return deviceCamera;
}

