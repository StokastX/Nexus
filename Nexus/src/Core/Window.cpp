#include "Window.h"
#include <cassert>
#include "Events/ApplicationEvent.h"
#include "Events/KeyEvent.h"
#include "Events/MouseEvent.h"

namespace Nexus {

	Window::Window(const WindowSpecification& specification)
		: m_Specification(specification) { }

	Window::~Window()
	{
		Destroy();
	}

	void Window::Create()
	{
		if (m_Specification.startMaximized)
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

		glfwSetWindowUserPointer(m_Handle, this);

		// Set GLFW callbacks
		glfwSetWindowSizeCallback(m_Handle, [](GLFWwindow* window, int width, int height)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			win->m_Specification.width = width;
			win->m_Specification.height = height;

			win->m_EventCallback(std::make_unique<WindowResizeEvent>(width, height));
		});

		glfwSetWindowCloseCallback(m_Handle, [](GLFWwindow* window)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			win->m_EventCallback(std::make_unique<WindowCloseEvent>());
		});

		glfwSetKeyCallback(m_Handle, [](GLFWwindow* window, int key, int scancode, int action, int mods)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);

			switch (action)
			{
				case GLFW_PRESS:
				{
					win->m_EventCallback(std::make_unique<KeyPressedEvent>(key, false));
					break;
				}
				case GLFW_RELEASE:
				{
					win->m_EventCallback(std::make_unique<KeyReleasedEvent>(key));
					break;
				}
				case GLFW_REPEAT:
				{
					win->m_EventCallback(std::make_unique<KeyPressedEvent>(key, true));
					break;
				}
			}
		});

		glfwSetCharCallback(m_Handle, [](GLFWwindow* window, unsigned int keycode)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			win->m_EventCallback(std::make_unique<KeyTypedEvent>(keycode));
		});

		glfwSetMouseButtonCallback(m_Handle, [](GLFWwindow* window, int button, int action, int mods)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);

			switch (action)
			{
				case GLFW_PRESS:
				{
					win->m_EventCallback(std::make_unique<MouseButtonPressedEvent>(button));
					break;
				}
				case GLFW_RELEASE:
				{
					win->m_EventCallback(std::make_unique<MouseButtonReleasedEvent>(button));
					break;
				}
			}
		});

		glfwSetScrollCallback(m_Handle, [](GLFWwindow* window, double xOffset, double yOffset)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			win->m_EventCallback(std::make_unique<MouseScrolledEvent>((float)xOffset, (float)yOffset));
		});

		glfwSetCursorPosCallback(m_Handle, [](GLFWwindow* window, double xPos, double yPos)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			win->m_EventCallback(std::make_unique<MouseMovedEvent>((float)xPos, (float)yPos));
		});
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
}