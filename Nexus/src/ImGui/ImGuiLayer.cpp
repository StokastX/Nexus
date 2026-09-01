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

	ImFont* ImGuiLayer::s_SemiBoldFont = nullptr;

	namespace {

		/*
		 * The editor palette is built from one accent colour. Every widget that means "active" or
		 * "selected" derives from it, so changing Accent re-tints the whole UI without touching any
		 * other colour. Grabs are the deliberate exception: see SliderGrab below.
		 */
		constexpr ImVec4 Accent = ImVec4(0.95f, 0.55f, 0.19f, 1.00f);

		/*
		 * Neutral ramp. The greys the theme already used, laid out as an ordered ladder so every
		 * surface picks a level instead of inventing one -- that ordering is what makes panels read
		 * as inset or raised relative to each other. Strictly neutral (R == G == B): any tint here
		 * shows up as a colour cast across the whole editor.
		 */
		constexpr ImVec4 Grey900 = ImVec4(0.086f, 0.086f, 0.086f, 1.00f);	// inset: tab bar, menu bar, empty dockspace
		constexpr ImVec4 Grey800 = ImVec4(0.118f, 0.118f, 0.118f, 1.00f);	// frames, buttons, title bars
		constexpr ImVec4 Grey700 = ImVec4(0.153f, 0.153f, 0.153f, 1.00f);	// hovered frames and buttons
		constexpr ImVec4 Grey600 = ImVec4(0.180f, 0.180f, 0.180f, 1.00f);	// window background
		constexpr ImVec4 Grey500 = ImVec4(0.220f, 0.220f, 0.220f, 1.00f);	// popups, menus
		constexpr ImVec4 Grey400 = ImVec4(0.290f, 0.290f, 0.290f, 1.00f);	// borders, grabs
		constexpr ImVec4 Grey300 = ImVec4(0.380f, 0.380f, 0.380f, 1.00f);	// hovered and held grabs

		constexpr ImVec4 TextBright = ImVec4(0.92f, 0.92f, 0.92f, 1.00f);
		constexpr ImVec4 TextFaint = ImVec4(0.42f, 0.42f, 0.42f, 1.00f);

		ImVec4 Alpha(const ImVec4& color, float alpha)
		{
			return ImVec4(color.x, color.y, color.z, alpha);
		}

		// Blends towards the accent. Used where a widget should read as tinted rather than coloured.
		ImVec4 Mix(const ImVec4& a, const ImVec4& b, float t)
		{
			return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
		}

	}

	ImGuiLayer::ImGuiLayer()
	{ }

	void ImGuiLayer::OnAttach()
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

		ApplyTheme();

		float xscale, yscale;
		glfwGetWindowContentScale(Application::Get().GetWindow().GetHandle(), &xscale, &yscale);

		// The first face added becomes the default the whole UI draws with. The second is pushed
		// explicitly around section headings; see UI::BeginSection.
		io.Fonts->AddFontFromFileTTF(Paths::Resolve("assets/fonts/Inter-Regular.otf").c_str(), 14.0f * xscale);
		s_SemiBoldFont = io.Fonts->AddFontFromFileTTF(Paths::Resolve("assets/fonts/Inter-SemiBold.otf").c_str(), 14.0f * xscale);

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

	void ImGuiLayer::ApplyTheme()
	{
		ImVec4* colors = ImGui::GetStyle().Colors;

		colors[ImGuiCol_Text] = TextBright;
		colors[ImGuiCol_TextDisabled] = TextFaint;

		colors[ImGuiCol_WindowBg] = Grey600;
		colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_PopupBg] = Alpha(Grey500, 0.98f);

		colors[ImGuiCol_Border] = Alpha(Grey400, 0.60f);
		colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

		// Inputs sit below the window surface, and pick up the accent while they are being edited.
		colors[ImGuiCol_FrameBg] = Grey800;
		colors[ImGuiCol_FrameBgHovered] = Grey700;
		colors[ImGuiCol_FrameBgActive] = Mix(Grey800, Accent, 0.14f);

		colors[ImGuiCol_TitleBg] = Grey900;
		colors[ImGuiCol_TitleBgActive] = Grey900;
		colors[ImGuiCol_TitleBgCollapsed] = Alpha(Grey900, 0.75f);

		// A step darker than the dockspace so the menu bar reads as its own strip.
		colors[ImGuiCol_MenuBarBg] = Grey900;

		colors[ImGuiCol_ScrollbarBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_ScrollbarGrab] = Alpha(Grey400, 0.70f);
		colors[ImGuiCol_ScrollbarGrabHovered] = Grey300;
		colors[ImGuiCol_ScrollbarGrabActive] = Grey300;

		colors[ImGuiCol_CheckMark] = Accent;

		// Grabs stay neutral. They are handles you drag, not state -- an accented slider cursor
		// reads as a value being highlighted, and with one per row the panel turns into confetti.
		// The frame tinting under FrameBgActive is what signals "being edited" instead.
		colors[ImGuiCol_SliderGrab] = Alpha(Grey400, 0.90f);
		colors[ImGuiCol_SliderGrabActive] = Grey300;

		colors[ImGuiCol_Button] = Grey800;
		colors[ImGuiCol_ButtonHovered] = Grey700;
		colors[ImGuiCol_ButtonActive] = Mix(Grey800, Accent, 0.30f);

		// Header is shared by three things: selected tree rows, Selectables, and the fill of framed
		// tree nodes / CollapsingHeaders -- and that last one is painted unconditionally, not only
		// when selected. Any accent here turns every section bar in the Properties panel into a
		// coloured block, so it stays neutral. Monotonic on purpose: hover is brighter than a
		// resting selection, held is brightest.
		colors[ImGuiCol_Header] = Grey500;
		colors[ImGuiCol_HeaderHovered] = Grey400;
		colors[ImGuiCol_HeaderActive] = Grey300;

		colors[ImGuiCol_Separator] = Alpha(Grey400, 0.55f);
		colors[ImGuiCol_SeparatorHovered] = Alpha(Accent, 0.60f);
		colors[ImGuiCol_SeparatorActive] = Accent;

		colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_ResizeGripHovered] = Alpha(Accent, 0.50f);
		colors[ImGuiCol_ResizeGripActive] = Accent;

		// The active tab takes the window colour so it merges into the panel it belongs to, while
		// the rest of the strip stays at the inset level.
		colors[ImGuiCol_Tab] = Grey900;
		colors[ImGuiCol_TabHovered] = Grey700;
		colors[ImGuiCol_TabActive] = Grey600;
		colors[ImGuiCol_TabUnfocused] = Grey900;
		colors[ImGuiCol_TabUnfocusedActive] = Mix(Grey900, Grey600, 0.60f);

		colors[ImGuiCol_DockingPreview] = Alpha(Accent, 0.35f);
		colors[ImGuiCol_DockingEmptyBg] = Grey900;

		colors[ImGuiCol_PlotLines] = Accent;
		colors[ImGuiCol_PlotLinesHovered] = Mix(Accent, TextBright, 0.35f);
		colors[ImGuiCol_PlotHistogram] = Accent;
		colors[ImGuiCol_PlotHistogramHovered] = Mix(Accent, TextBright, 0.35f);

		colors[ImGuiCol_TableHeaderBg] = Grey900;
		colors[ImGuiCol_TableBorderStrong] = Alpha(Grey400, 0.60f);
		colors[ImGuiCol_TableBorderLight] = Alpha(Grey400, 0.30f);
		colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
		colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);

		colors[ImGuiCol_TextSelectedBg] = Alpha(Accent, 0.35f);

		colors[ImGuiCol_DragDropTarget] = Accent;

		colors[ImGuiCol_NavHighlight] = Alpha(Accent, 0.90f);
		colors[ImGuiCol_NavWindowingHighlight] = Alpha(TextBright, 0.70f);
		colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.35f);

		colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.55f);

		ImGuiStyle& style = ImGui::GetStyle();

		style.WindowPadding = ImVec2(10.00f, 8.00f);
		// Vertical padding stays at the original 3: the extra pixel per widget was multiplied by
		// every row of every property table and made the panels noticeably airier than they were.
		style.FramePadding = ImVec2(7.00f, 3.00f);
		style.CellPadding = ImVec2(6.00f, 6.00f);
		style.ItemSpacing = ImVec2(8.00f, 6.00f);
		style.ItemInnerSpacing = ImVec2(6.00f, 4.00f);
		style.IndentSpacing = 14.0f;
		style.ScrollbarSize = 11.0f;
		style.GrabMinSize = 10.0f;
		style.DockingSeparatorSize = 2.0f;

		// A hairline on every surface.
		style.WindowBorderSize = 1.0f;
		style.ChildBorderSize = 1.0f;
		style.PopupBorderSize = 1.0f;
		style.FrameBorderSize = 1.0f;
		style.TabBorderSize = 0.0f;

		style.WindowRounding = 6.0f;
		style.ChildRounding = 4.0f;
		style.FrameRounding = 4.0f;
		style.PopupRounding = 6.0f;
		style.ScrollbarRounding = 4.0f;
		style.GrabRounding = 3.0f;
		style.TabRounding = 4.0f;

		style.WindowTitleAlign = ImVec2(0.00f, 0.50f);
		style.WindowMenuButtonPosition = ImGuiDir_None;
		style.SeparatorTextBorderSize = 1.0f;
		style.SeparatorTextAlign = ImVec2(0.00f, 0.50f);
		style.SeparatorTextPadding = ImVec2(0.00f, 8.00f);
	}

}
