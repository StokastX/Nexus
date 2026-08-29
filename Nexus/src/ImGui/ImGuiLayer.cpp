#include "ImGuiLayer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "Core/Application.h"
#include "Utils/Paths.h"

namespace Nexus {

	ImGuiLayer::ImGuiLayer()
	{ }

	void ImGuiLayer::OnAttach()
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

		ImGui::StyleColorsCustomDark();

		float xscale, yscale;
		glfwGetWindowContentScale(Application::Get().GetWindow().GetHandle(), &xscale, &yscale);

		io.Fonts->AddFontFromFileTTF(Paths::Resolve("assets/fonts/SF-Pro-Text-Regular.otf").c_str(), 14.0f * xscale);
		ImGui::GetStyle().ScaleAllSizes(xscale);

		ImGui_ImplGlfw_InitForOpenGL(Application::Get().GetWindow().GetHandle(), true);
		ImGui_ImplOpenGL3_Init("#version 460");
	}

	void ImGuiLayer::OnDetach()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	void ImGuiLayer::OnEvent(Event& e)
	{
		if (m_BlockEvents)
		{
			ImGuiIO& io = ImGui::GetIO();
			e.handled |= e.IsInCategory(EventCategoryMouse) & io.WantCaptureMouse;
			e.handled |= e.IsInCategory(EventCategoryKeyboard) & io.WantCaptureKeyboard;
		}
	}

	void ImGuiLayer::Begin()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	void ImGuiLayer::End()
	{
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	void ImGuiLayer::SetDarkThemeColors()
	{
		ImVec4* colors = ImGui::GetStyle().Colors;
		colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
		colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);

		colors[ImGuiCol_WindowBg] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
		colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_PopupBg] = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);

		colors[ImGuiCol_Border] = ImVec4(0.0f, 0.0f, 0.0f, 0.29f);
		colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);

		colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.0f);
		colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.12f, 0.12f, 0.54f);
		colors[ImGuiCol_FrameBgActive] = ImVec4(0.12f, 0.12f, 0.12f, 0.67f);

		colors[ImGuiCol_TitleBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
		colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
		colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.12f, 0.12f, 0.12f, 0.51f);

		colors[ImGuiCol_MenuBarBg] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);

		//colors[ImGuiCol_ScrollbarBg] = ImVec4(0.12f, 0.12f, 0.12f, 0.53f);
		colors[ImGuiCol_ScrollbarBg] = ImVec4(0.18f, 0.18f, 0.18f, 1.00);
		colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.30f, 0.30f, 0.30f, 0.80f);
		colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.38f, 0.38f, 0.38f, 0.80f);
		colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);

		colors[ImGuiCol_CheckMark] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);

		colors[ImGuiCol_SliderGrab] = ImVec4(0.30f, 0.30f, 0.30f, 0.80f);
		colors[ImGuiCol_SliderGrabActive] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);

		colors[ImGuiCol_Button] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
		colors[ImGuiCol_ButtonHovered] = ImVec4(0.12f, 0.12f, 0.12f, 0.54f);
		colors[ImGuiCol_ButtonActive] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);

		colors[ImGuiCol_Header] = ImVec4(0.30f, 0.30f, 0.30f, 0.72f);
		colors[ImGuiCol_HeaderHovered] = ImVec4(0.30f, 0.30f, 0.30f, 0.80f);
		colors[ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.30f, 0.30f, 0.80f);

		colors[ImGuiCol_Separator] = ImVec4(0.30f, 0.30f, 0.30f, 0.62f);
		colors[ImGuiCol_SeparatorHovered] = ImVec4(0.19f, 0.19f, 0.19f, 0.67f);
		colors[ImGuiCol_SeparatorActive] = ImVec4(0.19f, 0.19f, 0.19f, 1.00f);

		colors[ImGuiCol_ResizeGrip] = ImVec4(0.30f, 0.30f, 0.30f, 0.62f);
		colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
		colors[ImGuiCol_ResizeGripActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);

		colors[ImGuiCol_Tab] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_TabHovered] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
		colors[ImGuiCol_TabActive] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
		colors[ImGuiCol_TabUnfocused] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);

		colors[ImGuiCol_DockingPreview] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
		colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);

		colors[ImGuiCol_PlotLines] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
		colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
		colors[ImGuiCol_PlotHistogram] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
		colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);

		colors[ImGuiCol_TableHeaderBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
		colors[ImGuiCol_TableBorderStrong] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
		colors[ImGuiCol_TableBorderLight] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
		colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);

		colors[ImGuiCol_TextSelectedBg] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);

		colors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);

		colors[ImGuiCol_NavHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
		colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
		colors[ImGuiCol_NavWindowingDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);

		colors[ImGuiCol_ModalWindowDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);

		ImGuiStyle& style = ImGui::GetStyle();
		style.WindowPadding = ImVec2(8.00f, 8.00f);
		style.FramePadding = ImVec2(5.00f, 3.00f);
		style.CellPadding = ImVec2(6.00f, 6.00f);
		style.ItemSpacing = ImVec2(6.00f, 6.00f);
		//style.ItemInnerSpacing = ImVec2(6.00f, 6.00f);
		//style.TouchExtraPadding = ImVec2(0.00f, 0.00f);
		style.DockingSeparatorSize = 3;
		style.IndentSpacing = 12;
		style.ScrollbarSize = 10;
		style.GrabMinSize = 10;
		style.WindowBorderSize = 1;
		style.ChildBorderSize = 1;
		//style.PopupBorderSize = 1;
		//style.FrameBorderSize = 1;
		//style.TabBorderSize = 1;
		style.TabRounding = 4;
		style.WindowRounding = 4;
		//style.ChildRounding = 4;
		style.FrameRounding = 4;
		style.PopupRounding = 4;
		//style.ScrollbarRounding = 9;
		style.GrabRounding = 4;
		//style.LogSliderDeadzone = 4;
		//style.TabRounding = 4;
		style.SeparatorTextBorderSize = 4;
		style.WindowMenuButtonPosition = ImGuiDir_None;
	}

}
