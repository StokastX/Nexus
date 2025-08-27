#pragma once

#include "Renderer/Renderer.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "Input.h"

class ViewportPanel
{
public:
	ViewportPanel(Renderer* renderer);
	void OnImGuiRender(bool fitRenderToViewport);
	ImVec2 GetViewportSize() { return m_ViewportSize; }

private:
	Renderer* m_Renderer;
	float m_RenderScale = 1.0f;
	ImVec2 m_RenderScroll = ImVec2(0.0f, 0.0f);
	ImVec2 m_TopLeft = ImVec2(0.0f, 0.0f);
	ImVec2 m_ViewportSize = ImVec2(0.0f, 0.0f);
	bool initialized = false;
};