#include "Application.h"

#include <cassert>
#include "Core/Input.h"
#include "ImGui/ImGuiLayer.h"
#include "Events/ApplicationEvent.h"


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

		m_Window->SetEventCallback([this](Event& e) {
			this->OnEvent(e);
		});

		PushLayer<ImGuiLayer>();
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

	void Application::OnEvent(Event& e)
	{
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>([this](WindowCloseEvent& event) {
			m_Running = false;
			return true;
			});

		dispatcher.Dispatch<WindowResizeEvent>([this](WindowResizeEvent& event) {
			if (event.GetWidth() == 0 || event.GetHeight() == 0)
			{
				m_Minimized = true;
				return false;
			}
			m_Minimized = false;
			return false;
			});

		for (auto it = m_LayerStack.rbegin(); it != m_LayerStack.rend(); it++)
		{
			if (e.handled)
				break;
			(*it)->OnEvent(e);
		}
	}

	void Application::Run()
	{
		m_Running = true;

		float lastTime = GetTime();

		// Main Application loop
		while (m_Running)
		{
			glfwPollEvents();

			float currentTime = GetTime();
			float timestep = (currentTime - lastTime) * 1000.0f;
			lastTime = currentTime;

			// Main layer update
			for (const std::unique_ptr<Layer>& layer : m_LayerStack)
				layer->OnUpdate(timestep);

			ImGuiLayer* imGuiLayer = static_cast<ImGuiLayer*>(m_LayerStack[0].get());
			imGuiLayer->Begin();

			// Layer render
			for (const std::unique_ptr<Layer>& layer : m_LayerStack)
				layer->OnRender();

			imGuiLayer->End();

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
