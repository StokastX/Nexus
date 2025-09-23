#include "Window.h"
#include <cassert>

namespace Nexus {

	Window::Window(const WindowSpecification& specification)
		: m_Specification(specification) { }

	Window::~Window()
	{
		Destroy();
	}

	void Window::Create()
	{
		glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		m_Handle = glfwCreateWindow(m_Specification.width, m_Specification.height, m_Specification.title.c_str(), nullptr, nullptr);
		if (!m_Handle)
		{
			std::cout << "Window: Error creating glfw window" << std::endl;
			glfwTerminate();
			assert(false);
		}

		glfwMakeContextCurrent(m_Handle);

		// vsync (frame rate / screen refresh rate synchronization)
		glfwSwapInterval(m_Specification.vSync ? 1 : 0);

		if (glewInit() != GLEW_OK)
			std::cout << "Window: Error initializing GLEW" << std::endl;
	}

	void Window::Destroy()
	{
		if (m_Handle)
			glfwDestroyWindow(m_Handle);

		m_Handle = nullptr;
	}

	void Window::Update()
	{
		glfwSwapBuffers(m_Handle);
	}

	int2 Window::GetFramebufferSize()
	{
		int2 size;
		glfwGetFramebufferSize(m_Handle, &size.x, &size.y);
		return size;
	}

	bool Window::ShouldClose() const
	{
		return glfwWindowShouldClose(m_Handle) != 0;
	}
}