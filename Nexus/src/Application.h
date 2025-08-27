#pragma once
//#include "Renderer/Renderer.h"
#include "Renderer/Panels/ViewportPanel.h"
#include "Renderer/Panels/MetricsPanel.h"
#include "Renderer/Panels/SceneHierarchyPanel.h"
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
	Scene m_Scene;
	ViewportPanel m_ViewportPanel;
	SceneHierarchyPanel m_SceneHierarchyPanel;
	MetricsPanel m_MetricsPanel;

	static GLFWwindow* m_Window;
};
