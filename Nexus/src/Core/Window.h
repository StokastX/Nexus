#pragma once

#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Utils/cuda_math.h"

namespace Nexus {

	struct WindowSpecification
	{
		std::string title;
		uint32_t width = 1920;
		uint32_t height = 1080;
		bool IsResizable = true;
		bool vSync = true;
		bool startMaximized = false;
	};

	class Window
	{
	public:
		Window(const WindowSpecification& specification = WindowSpecification());
		~Window();

		void Create();
		void Destroy();

		void Update();

		int2 GetFramebufferSize();

		bool ShouldClose() const;
		GLFWwindow* GetHandle() const { return m_Handle; }

	private:
		WindowSpecification m_Specification;

		GLFWwindow* m_Handle = nullptr;
	};

}