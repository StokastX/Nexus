#include "Window.h"
#include <cassert>
#include <cuda_gl_interop.h>
#include "Utils/Utils.h"
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

		BindCudaToContextDevice();

		glfwSetWindowUserPointer(m_Handle, this);

		// Set GLFW callbacks
		glfwSetWindowSizeCallback(m_Handle, [](GLFWwindow* window, int width, int height)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			win->m_Specification.width = width;
			win->m_Specification.height = height;

			WindowResizeEvent event(width, height);
			win->m_EventCallback(event);
		});

		glfwSetWindowCloseCallback(m_Handle, [](GLFWwindow* window)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			WindowCloseEvent event;
			win->m_EventCallback(event);
		});

		glfwSetKeyCallback(m_Handle, [](GLFWwindow* window, int key, int scancode, int action, int mods)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);

			switch (action)
			{
				case GLFW_PRESS:
				{
					KeyPressedEvent event(key, false);
					win->m_EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					KeyReleasedEvent event(key);
					win->m_EventCallback(event);
					break;
				}
				case GLFW_REPEAT:
				{
					KeyPressedEvent event(key, true);
					win->m_EventCallback(event);
					break;
				}
			}
		});

		glfwSetCharCallback(m_Handle, [](GLFWwindow* window, unsigned int keycode)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			KeyTypedEvent event(keycode);
			win->m_EventCallback(event);
		});

		glfwSetMouseButtonCallback(m_Handle, [](GLFWwindow* window, int button, int action, int mods)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);

			switch (action)
			{
				case GLFW_PRESS:
				{
					MouseButtonPressedEvent event(button);
					win->m_EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					MouseButtonReleasedEvent event(button);
					win->m_EventCallback(event);
					break;
				}
			}
		});

		glfwSetScrollCallback(m_Handle, [](GLFWwindow* window, double xOffset, double yOffset)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			MouseScrolledEvent event((float)xOffset, (float)yOffset);
			win->m_EventCallback(event);
		});

		glfwSetCursorPosCallback(m_Handle, [](GLFWwindow* window, double xPos, double yPos)
		{
			Window* win = (Window*)glfwGetWindowUserPointer(window);
			MouseMovedEvent event((float)xPos, (float)yPos);
			win->m_EventCallback(event);
		});
	}

	void Window::Destroy()
	{
		if (m_Handle)
			glfwDestroyWindow(m_Handle);

		m_Handle = nullptr;
	}


	// CUDA-OpenGL interop only works when the GL context and the CUDA context are on the
	// same physical device. Bind CUDA to whichever device is actually driving the context,
	// and fail loudly rather than letting cudaGraphicsGLRegisterImage() report a bare
	// cudaErrorOperatingSystem (304) later on.
	void Window::BindCudaToContextDevice()
	{
		unsigned int deviceCount = 0;
		int devices[8] = {};
		cudaError_t result = cudaGLGetDevices(&deviceCount, devices, 8, cudaGLDeviceListAll);
		cudaGetLastError();

		if (result != cudaSuccess || deviceCount == 0)
		{
			std::cerr << "Window: the OpenGL context is running on '" << glGetString(GL_RENDERER)
				<< "', which is not a CUDA device. CUDA-OpenGL interop cannot work." << std::endl
				<< "        On a hybrid-graphics laptop, force this executable onto the NVIDIA GPU "
				<< "(Windows Settings > System > Display > Graphics, or the NVIDIA Control Panel)."
				<< std::endl;
			assert(false);
			return;
		}

		CheckCudaErrors(cudaSetDevice(devices[0]));
	}

	void Window::Update()
	{
		glfwPollEvents();
		glfwSwapBuffers(m_Handle);
	}

	int2 Window::GetFramebufferSize()
	{
		int2 size;
		glfwGetFramebufferSize(m_Handle, &size.x, &size.y);
		return size;
	}
}