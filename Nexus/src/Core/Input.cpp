#include "Input.h"
#include "Core/Application.h"
#include <iostream>

namespace Nexus {

	void Input::Init(GLFWwindow* window)
	{
		glfwSetScrollCallback(window, ScrollCallback);
	}

	float2 Input::GetMousePosition()
	{
		double xpos, ypos;
		glfwGetCursorPos(Application::Get().GetWindow().GetHandle(), &xpos, &ypos);
		return make_float2(xpos, ypos);
	}

	bool Input::IsKeyDown(int key)
	{
		int state = glfwGetKey(Application::Get().GetWindow().GetHandle(), key);
		return state == GLFW_PRESS;
	}

	bool Input::IsMouseButtonDown(int key)
	{
		int state = glfwGetMouseButton(Application::Get().GetWindow().GetHandle(), key);
		return state == GLFW_PRESS;
	}

	void Input::SetCursorMode(int mode)
	{
		glfwSetInputMode(Application::Get().GetWindow().GetHandle(), GLFW_CURSOR, mode);
	}

	void Input::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		// Store only Y offset (vertical scroll)
		if (IsKeyDown(GLFW_KEY_LEFT_CONTROL))
			m_ScrollOffsetY = (float)yoffset;
	}

	float Input::GetScrollOffsetY()
	{
		float offset = m_ScrollOffsetY;
		// Reset so it’s only reported once per frame
		m_ScrollOffsetY = 0.0f;
		return offset;
	}

}
