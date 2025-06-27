#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "Renderer.h"
#include "Utils/Utils.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "FileDialog.h"


Renderer::Renderer(uint32_t width, uint32_t height, GLFWwindow* window, Scene* scene)
	: m_ViewportWidth(width), m_ViewportHeight(height), m_Scene(scene), m_HierarchyPannel(scene),
	m_MetricsPanel(scene), m_RenderTexture(width, height), m_PathTracer(width, height)
{
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

Renderer::~Renderer()
{
	ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Renderer::Reset()
{
	m_PathTracer.ResetFrameNumber();
	m_MetricsPanel.Reset();
}

void Renderer::Render(Scene& scene, float deltaTime)
{ 
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	m_MetricsPanel.UpdateMetrics(deltaTime);

	// Position UI and resize the texture and pixel buffer depending on the viewport size
	RenderUI(scene);

	if (scene.IsInvalid())
	{
		scene.Update();
		m_PathTracer.ResetFrameNumber();
	}

	// Launch cuda path tracing kernel, writes the viewport into the pixelbuffer
	if (!scene.IsEmpty())
	{
		m_PathTracer.UpdateDeviceScene(*m_Scene);
		m_PathTracer.Render(scene);

		// Unpack the pixel buffer written by cuda to the renderer texture
		UnpackToTexture();
	}
	else
		m_PathTracer.ResetFrameNumber();


	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Renderer::RenderUI(Scene& scene)
{
	ImGui::DockSpaceOverViewport();

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Open...", "Ctrl+O"))
			{
				std::string fullPath = FileDialog::OpenFile(
					"3D model (*.obj;*.ply;*.stl;*.glb;*.gltf;*.fbx;*.3ds;*.max;*.blend;*.dae)\0*.obj;*.ply;*.stl;*.glb;*.gltf;*.fbx;*.3ds;*.max;*.x3d;*.blend;*.dae\0"
				);
				if (!fullPath.empty())
				{
					CheckCudaErrors(cudaDeviceSynchronize());
					Reset();
					scene.Reset();

					std::string fileName, filePath;
					Utils::GetPathAndFileName(fullPath, filePath, fileName);
					scene.CreateMeshInstanceFromFile(filePath, fileName);
					CheckCudaErrors(cudaDeviceSynchronize());
				}
			}

			if (ImGui::MenuItem("Load HDR map", "Ctrl+H"))
			{
				std::string fullPath = FileDialog::OpenFile(
					"HDR file (*.hdr)\0*.hdr;*.exr\0"
				);
				if (!fullPath.empty())
				{
					std::string fileName, filePath;
					Utils::GetPathAndFileName(fullPath, filePath, fileName);
					scene.AddHDRMap(filePath, fileName);
					m_PathTracer.ResetFrameNumber();
				}
			}

			if (ImGui::MenuItem("Save Screenshot", "Ctrl+S")) {
				SaveScreenshot();
			}

			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("Viewport");

	if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !scene.IsEmpty())
	{
		ImVec2 viewportPos = ImGui::GetCursorScreenPos();
		ImVec2 mousePos = ImGui::GetMousePos();
		int2 hoveredPixel = make_int2(mousePos.x - viewportPos.x, mousePos.y - viewportPos.y);
		hoveredPixel.y = m_ViewportHeight - hoveredPixel.y;

		if (hoveredPixel.x >= 0 && hoveredPixel.x < m_ViewportWidth && hoveredPixel.y >= 0 && hoveredPixel.y < m_ViewportHeight)
		{
			m_PathTracer.SetPixelQuery(hoveredPixel.x, hoveredPixel.y);
		}
	}


	uint32_t viewportWidth = ImGui::GetContentRegionAvail().x;
	uint32_t viewportHeight = ImGui::GetContentRegionAvail().y;

	scene.GetCamera()->OnResize(viewportWidth, viewportHeight);
	OnResize(viewportWidth, viewportHeight);

	ImGui::Image((void *)(intptr_t)m_RenderTexture.GetHandle(), ImVec2(m_RenderTexture.GetWidth(), m_RenderTexture.GetHeight()), ImVec2(0, 1), ImVec2(1, 0));

	ImGui::End();
	ImGui::PopStyleVar();
	
	m_HierarchyPannel.SetSelectionContext(m_PathTracer.GetSelectedInstance());
	m_HierarchyPannel.OnImGuiRender();

	m_MetricsPanel.OnImGuiRender(m_PathTracer.GetFrameNumber());
}

void Renderer::UnpackToTexture()
{
	m_RenderTexture.Bind();
	const PixelBuffer& pixelBuffer = m_PathTracer.GetPixelBuffer();
	pixelBuffer.Bind();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_RenderTexture.GetWidth(), m_RenderTexture.GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
	pixelBuffer.Unbind();
}

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if ((m_ViewportWidth != width || m_ViewportHeight != height) && width != 0 && height != 0)
	{
		m_PathTracer.OnResize(width, height);
		m_MetricsPanel.Reset();
		m_RenderTexture.OnResize(width, height);
		m_ViewportWidth = width;
		m_ViewportHeight = height;
	}
}

void Renderer::SaveScreenshot()
{
	int width = m_RenderTexture.GetWidth();
	int height = m_RenderTexture.GetHeight();
	std::vector<unsigned char> pixels(width * height * 4);

	glBindTexture(GL_TEXTURE_2D, m_RenderTexture.GetHandle());
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
	glBindTexture(GL_TEXTURE_2D, 0);

	stbi_flip_vertically_on_write(1);

	std::string filepath = FileDialog::SaveFile(
		"PNG image (*.png)\0*.png\0"
	);

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
