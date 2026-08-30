#pragma once

#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <functional>
#include "Utils/cuda_math.h"
#include "Events/Event.h"

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
		using EventCallbackFn = std::function<void(Event& event)>;

	public:
		Window(const WindowSpecification& specification = WindowSpecification());
		~Window();

		void SetEventCallback(const EventCallbackFn& callback) { m_EventCallback = callback; }

		void Create();
		void Destroy();

		void Update();

		int2 GetFramebufferSize();

		GLFWwindow* GetHandle() const { return m_Handle; }

	private:
		void BindCudaToContextDevice();

		WindowSpecification m_Specification;
		GLFWwindow* m_Handle = nullptr;

		EventCallbackFn m_EventCallback;
	};

}