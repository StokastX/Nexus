#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <queue>

#include "Window.h"
#include "Layer.h"
#include "Utils/cuda_math.h"

namespace Nexus {

	struct ApplicationSpecification
	{
		std::string name = "Application";
		WindowSpecification windowSpec;
	};

	class Application {

	public:
		Application(const ApplicationSpecification& specification = ApplicationSpecification());
		~Application();

		void OnEvent(std::unique_ptr<Event> event);

		void Run();
		void Stop();

		template <typename TLayer>
		void PushLayer()
		{
			static_assert(std::is_base_of<Layer, TLayer>::value, "TLayer must derive from Layer");
			m_LayerStack.push_back(std::make_unique<TLayer>());
			m_LayerStack.back()->OnAttach();
		}

		int2 GetFramebufferSize() const;

		Window& GetWindow() { return *m_Window; }

		static Application& Get();
		static float GetTime();

	private:
		void DispatchEvents();

	private:
		ApplicationSpecification m_Specification;
		std::shared_ptr<Window> m_Window;
		bool m_Running = false;
		bool m_Minimized = false;

		std::vector<std::unique_ptr<Layer>> m_LayerStack;
		std::queue<std::unique_ptr<Event>> m_EventQueue;
	};

}
