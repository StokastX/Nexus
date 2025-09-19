#pragma once
#include "Renderer/Panels/RenderPanel.h"
#include "Renderer/Panels/ViewportPanel.h"
#include "Renderer/Panels/MetricsPanel.h"
#include "Renderer/Panels/SceneHierarchyPanel.h"
#include "OpenGL/OGLRenderer.h"
#include "Scene/Scene.h"

class Application {

public:
	Application(int width, int height, GLFWwindow* window);
	~Application();

	void Update(float deltaTime);

	void Display(float deltaTime);
	void RenderUI();

	void OnResize(int width, int height);
	void SaveScreenshot();

	static GLFWwindow* GetNativeWindow() { return m_Window; }

private:
	Renderer m_Renderer;
	OGLRenderer m_OGLRenderer;

	Scene m_Scene;
	RenderPanel m_RenderPanel;
	ViewportPanel m_ViewportPanel;
	SceneHierarchyPanel m_SceneHierarchyPanel;
	MetricsPanel m_MetricsPanel;

	static GLFWwindow* m_Window;
};
