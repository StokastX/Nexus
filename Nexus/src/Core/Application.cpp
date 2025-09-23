#include "Application.h"

#include <cassert>
#include "Core/Input.h"


namespace Nexus {

	static Application* s_Application = nullptr;

	static void GLFWErrorCallback(int error, const char* description)
	{
		std::cerr << "[GLFW Error]: " << description << std::endl;
	}

	Application::Application(const ApplicationSpecification& specification)
		: m_Specification(specification)
	{
		s_Application = this;

		glfwSetErrorCallback(GLFWErrorCallback);
		if (!glfwInit())
		{
			std::cout << "Application: Error initializing glfw" << std::endl;
			assert(false);
		}

		// Set window title to app name if empty
		if (m_Specification.windowSpec.title.empty())
			m_Specification.windowSpec.title = m_Specification.name;

		m_Window = std::make_shared<Window>(m_Specification.windowSpec);
		m_Window->Create();

		Input::Init(m_Window->GetHandle());
	}

	Application::~Application()
	{
		for (const std::unique_ptr<Layer>& layer : m_LayerStack)
			layer->OnDetach();
		m_LayerStack.clear();

		m_Window->Destroy();

		glfwTerminate();

		s_Application = nullptr;
	}

	void Application::Run()
	{
		m_Running = true;

		float lastTime = GetTime();

		// Main Application loop
		while (m_Running)
		{
			glfwPollEvents();

			if (m_Window->ShouldClose())
			{
				Stop();
				break;
			}

			float currentTime = GetTime();
			float timestep = (currentTime - lastTime) * 1000.0f;
			lastTime = currentTime;

			// Main layer update
			for (const std::unique_ptr<Layer>& layer : m_LayerStack)
				layer->OnUpdate(timestep);

			// Layer render
			for (const std::unique_ptr<Layer>& layer : m_LayerStack)
				layer->OnRender();

			m_Window->Update();
		}
	}

	void Application::Stop()
	{
		m_Running = false;
	}

	int2 Application::GetFramebufferSize() const
	{
		return m_Window->GetFramebufferSize();
	}

	Application& Application::Get()
	{
		assert(s_Application);
		return *s_Application;
	}

	float Application::GetTime()
	{
		return (float)glfwGetTime();
	}

}
