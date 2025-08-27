#include "Input.h"
#include <iostream>

GLFWwindow* Input::m_Window;

void Input::Init(GLFWwindow* window)
{
	m_Window = window;
	glfwSetScrollCallback(window, ScrollCallback);
}

float2 Input::GetMousePosition()
{
	double xpos, ypos;
	glfwGetCursorPos(m_Window, &xpos, &ypos);
	return make_float2(xpos, ypos);
}

bool Input::IsKeyDown(int key)
{
	int state = glfwGetKey(m_Window, key);
	return state == GLFW_PRESS;
}

bool Input::IsMouseButtonDown(int key)
{
	int state = glfwGetMouseButton(m_Window, key);
	return state == GLFW_PRESS;
}

void Input::SetCursorMode(int mode)
{
	glfwSetInputMode(m_Window, GLFW_CURSOR, mode);
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
