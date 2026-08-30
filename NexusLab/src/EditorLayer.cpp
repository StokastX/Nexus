#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image_write.h>

#include "EditorLayer.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


#include "Renderer/FileDialog.h"
#include "Core/Application.h"

namespace Nexus {

	EditorLayer::EditorLayer()
		: m_Scene(), m_Renderer(&m_Scene),
		m_OGLRenderer(&m_Scene), m_SceneHierarchyPanel(&m_Scene), m_MetricsPanel(&m_Renderer),
		m_ViewportPanel(&m_OGLRenderer), m_RenderPanel(&m_Renderer)
	{
	}

	EditorLayer::~EditorLayer()
	{
	}

	void EditorLayer::OnUpdate(float deltaTime)
	{
		m_Scene.GetCamera()->OnUpdate(deltaTime);
		m_MetricsPanel.UpdateMetrics(deltaTime);
	}

	void EditorLayer::OnRender()
	{
		// Render UI
		RenderUI();

		if (m_RenderPanel.ExportRender())
			SaveScreenshot();

		// Render the scene
		if (m_RenderPanel.MustRender())
			m_Renderer.Render();

		m_OGLRenderer.Render(m_SceneHierarchyPanel.GetSelectionContext());

		// Update selection after the render pass
		if (m_OGLRenderer.PixelQueryPending())
			m_SceneHierarchyPanel.SetSelectionContext(SelectionContext::Type::INSTANCE, m_OGLRenderer.GetPixelQuery().instanceIdx);
		// For debugging purposes
		if (m_Renderer.GetPathTracer()->PixelQueryPending())
			m_Renderer.GetPathTracer()->SynchronizePixelQuery();
	}

	void EditorLayer::OnEvent(Event& e)
	{
		EventDispatcher dispatcher(e);

		dispatcher.Dispatch<KeyPressedEvent>([this](KeyPressedEvent& event) {

			if (event.GetKeyCode() == Key::O && Input::IsKeyPressed(Key::LeftControl))
			{
				LoadScene();
				return true;
			}
			else if (event.GetKeyCode() == Key::H && Input::IsKeyPressed(Key::LeftControl))
			{
				LoadHdrMap();
				return true;
			}

			return false;
		});
	}

	void EditorLayer::LoadScene()
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

	void EditorLayer::LoadHdrMap()
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

	void EditorLayer::RenderUI()
	{
		ImGui::DockSpaceOverViewport();

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 5.0f * ImGui::GetWindowDpiScale()));
		bool menuBar = ImGui::BeginMainMenuBar();
		ImGui::PopStyleVar();
		if (menuBar);
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Open...", "Ctrl+O"))
					LoadScene();

				if (ImGui::MenuItem("Load HDR map", "Ctrl+H"))
					LoadHdrMap();

				ImGui::EndMenu();
			}
			ImGui::EndMainMenuBar();
		}

		// Render ImGui panels
		m_ViewportPanel.OnImGuiRender();
		m_RenderPanel.OnImGuiRender(m_MetricsPanel.FitRenderToViewport());
		m_SceneHierarchyPanel.OnImGuiRender();
		m_MetricsPanel.OnImGuiRender(m_Renderer.GetPathTracer()->GetFrameNumber());
	}

	void EditorLayer::SaveScreenshot()
	{
		InteropTexture& renderTexture = m_Renderer.GetTexture();
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

}
