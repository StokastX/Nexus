#pragma once

#include <iostream>
#include "Core/Layer.h"

namespace Nexus {

	class ImGuiLayer : public Layer
	{
	public:
		ImGuiLayer();
		~ImGuiLayer() = default;

		virtual void OnAttach() override;
		virtual void OnDetach() override;

		virtual void OnEvent(Event& e) override;

		void Begin();
		void End();

		void SetDarkThemeColors();
	};

}