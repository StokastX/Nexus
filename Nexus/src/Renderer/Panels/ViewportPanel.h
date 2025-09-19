#pragma once

#include "OpenGL/OGLRenderer.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "Input.h"

class ViewportPanel
{
public:
	ViewportPanel(OGLRenderer* renderer);
	void OnImGuiRender();
	ImVec2 GetViewportSize() { return m_ViewportSize; }

private:
	OGLRenderer* m_Renderer;
	ImVec2 m_ViewportSize = ImVec2(0.0f, 0.0f);
};