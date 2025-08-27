#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "Application.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "Renderer/FileDialog.h"

GLFWwindow* Application::m_Window;

Application::Application(int width, int height, GLFWwindow *window)
	: m_Scene(make_uint2(width, height)), m_Renderer(make_uint2(width, height), &m_Scene),
	m_SceneHierarchyPanel(&m_Scene), m_MetricsPanel(&m_Renderer), m_ViewportPanel(&m_Renderer)
{
	m_Window = window;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	float xscale, yscale;
	glfwGetWindowContentScale(window, &xscale, &yscale);

	io.FontDefault = io.Fonts->AddFontFromFileTTF("assets/fonts/opensans/OpenSans-Regular.ttf", 16.0f * xscale);
	ImGui::GetStyle().ScaleAllSizes(xscale);

    ImGui::StyleColorsCustomDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 130");
}

Application::~Application()
{
	ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Application::Update(float deltaTime)
{
	m_Scene.GetCamera()->OnUpdate(deltaTime);
	Display(deltaTime);
}

void Application::Display(float deltaTime)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	m_MetricsPanel.UpdateMetrics(deltaTime);

	// Render UI
	RenderUI();

	// Render the scene
	m_Renderer.Render(m_Scene, deltaTime);

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	// Update selection after the render pass
	if (m_Renderer.GetPathTracer()->PixelQueryPending())
		m_SceneHierarchyPanel.SetSelectionContext(SelectionContext::Type::INSTANCE, m_Renderer.GetPathTracer()->SynchronizePixelQuery());
}

void Application::RenderUI()
{
	ImGui::DockSpaceOverViewport();

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Open...", "Ctrl+O"))
			{
				std::vector<const char*> filters = { "*.obj", "*.ply", "*.stl", "*.glb", "*.gltf", "*.fbx", "*.3ds" };
				std::string fullPath = FileDialog::OpenFile(filters, "Scene File");
				if (!fullPath.empty())
				{
					CheckCudaErrors(cudaDeviceSynchronize());
					m_Renderer.Reset();
					m_MetricsPanel.Reset();
					m_Scene.Reset();

					std::string fileName, filePath;
					Utils::GetPathAndFileName(fullPath, filePath, fileName);
					m_Scene.CreateMeshInstanceFromFile(filePath, fileName);
					CheckCudaErrors(cudaDeviceSynchronize());
				}
			}

			if (ImGui::MenuItem("Load HDR map", "Ctrl+H"))
			{
				std::vector<const char*> filters = { "*.hdr", "*.exr" };
				std::string fullPath = FileDialog::OpenFile(filters, "HDR File");
				if (!fullPath.empty())
				{
					std::string fileName, filePath;
					Utils::GetPathAndFileName(fullPath, filePath, fileName);
					m_Scene.AddHDRMap(filePath, fileName);
					m_Renderer.GetPathTracer()->ResetFrameNumber();
				}
			}

			if (ImGui::MenuItem("Save Screenshot", "Ctrl+S")) {
				SaveScreenshot();
			}

			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	// Render ImGui panels
	m_ViewportPanel.OnImGuiRender(m_MetricsPanel.FitRenderToViewport());
	m_SceneHierarchyPanel.OnImGuiRender();
	m_MetricsPanel.OnImGuiRender(m_Renderer.GetPathTracer()->GetFrameNumber(), m_ViewportPanel.GetViewportSize());
}

void Application::OnResize(int width, int height)
{
}

void Application::SaveScreenshot()
{
	OGLTexture renderTexture = m_Renderer.GetTexture();
	int width = renderTexture.GetWidth();
	int height = renderTexture.GetHeight();
	std::vector<unsigned char> pixels(width * height * 4);

	glBindTexture(GL_TEXTURE_2D, renderTexture.GetHandle());
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
	glBindTexture(GL_TEXTURE_2D, 0);

	stbi_flip_vertically_on_write(1);

	std::vector<const char*> filters = { "*.png" };
	std::string filepath = FileDialog::SaveFile(filters, "PNG Image");

	const std::string extension = ".png";

	if (!filepath.empty())
	{
		// Add extension if necessary
		if (filepath.length() < extension.length() ||
			filepath.compare(filepath.size() - extension.size(), extension.size(), extension) != 0)
			filepath += extension;

		if (!stbi_write_png(filepath.c_str(), width, height, 4, pixels.data(), width * 4))
		{
			std::cerr << "Failed to save screenshot to " << filepath << std::endl;
		}
	}

	std::cout << "Screenshot saved at: " << filepath.c_str() << std::endl;
}
