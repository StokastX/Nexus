#pragma once
#include <GL/glew.h>
#include "GLFW/glfw3.h"
#include "Utils/cuda_math.h"
#include "Events/KeyEvent.h"
#include "Events/MouseEvent.h"

namespace Nexus {

	class Input
	{
	public:
		static float2 GetMousePosition();
		static bool IsKeyPressed(KeyCode keycode);
		static bool IsMouseButtonPressed(MouseCode button);
		static void SetCursorMode(int mode);
		static float GetScrollOffsetY();


	private:
		static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

		static inline float m_ScrollOffsetY = 0.0f;
	};

}
