#pragma once
#define IMGUI_DEFINE_MATH_OPERATORS
#include "Core/Layer.h"
#include "Renderer/Renderer.h"
#include "OpenGL/OGLRenderer.h"
#include "Panels/MetricsPanel.h"
#include "Panels/SceneHierarchyPanel.h"
#include "Panels/ViewportPanel.h"
#include "Panels/RenderPanel.h"

namespace Nexus {

	class EditorLayer : public Layer
	{
	public:
		EditorLayer();
		virtual ~EditorLayer();

		virtual void OnAttach() override;
		virtual void OnUpdate(float deltaTime) override;
		virtual void OnRender() override;

		virtual void OnEvent(Event& e) override;

		void LoadScene();
		void LoadHdrMap();

	private:
		void RenderUI();
		void SaveScreenshot();

	private:
		Scene m_Scene;
		Renderer m_Renderer;
		OGLRenderer m_OGLRenderer;

		RenderPanel m_RenderPanel;
		ViewportPanel m_ViewportPanel;
		SceneHierarchyPanel m_SceneHierarchyPanel;
		MetricsPanel m_MetricsPanel;
	};

}