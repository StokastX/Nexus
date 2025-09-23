#pragma once

#include <string>
#include <memory>
#include <vector>
#include <set>

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

		void Run();
		void Stop();

		template <typename TLayer>
		//requires(std::is_base_of_v<Layer, TLayer>)
		void PushLayer()
		{
			m_LayerStack.push_back(std::make_unique<TLayer>());
			m_LayerStack.back()->OnAttach();
		}

		int2 GetFramebufferSize() const;

		Window& GetWindow() { return *m_Window; }

		static Application& Get();
		static float GetTime();


	private:
		ApplicationSpecification m_Specification;
		std::shared_ptr<Window> m_Window;
		bool m_Running = false;

		std::vector<std::unique_ptr<Layer>> m_LayerStack;
	};

}
