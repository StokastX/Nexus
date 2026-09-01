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

		void BlockEvents(bool block) { m_BlockEvents = block; }

		// Applies the editor palette and the style metrics. The palette is derived from a single
		// accent colour, selected at the top of ImGuiLayer.cpp. Called by OnAttach.
		//
		// This header deliberately does not include imgui.h. Application.h pulls it in, and several
		// translation units include imgui_internal.h behind their own IMGUI_DEFINE_MATH_OPERATORS,
		// which imgui_internal.h #errors on when imgui.h was already included without it.
		static void ApplyTheme();

	private:
		bool m_BlockEvents = true;
	};

}
