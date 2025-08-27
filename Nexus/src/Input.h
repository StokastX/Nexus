#pragma once
#include "GLFW/glfw3.h"
#include "Utils/cuda_math.h"

class Input
{
public:
	static void Init(GLFWwindow* window);
	static float2 GetMousePosition();
	static bool IsKeyDown(int key);
	static bool IsMouseButtonDown(int key);
	static void SetCursorMode(int mode);
	static float GetScrollOffsetY();


private:
	static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

	static inline float m_ScrollOffsetY = 0.0f;
	static GLFWwindow* m_Window;
};

