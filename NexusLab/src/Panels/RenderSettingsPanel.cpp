#include "RenderSettingsPanel.h"
#include "UI/Section.h"

#include <GLFW/glfw3.h>

namespace Nexus {

	RenderSettingsPanel::RenderSettingsPanel(Renderer* context) : m_Context(context)
	{
		Reset();
	}

	void RenderSettingsPanel::Reset()
	{
		m_NAccumulatedFrame = 0;
		m_AccumulatedTime = 0.0f;
		m_DeltaTime = 0.0f;

		m_MRPS = 0.0f;
		m_NumRaysProcessed = 0;

		m_DisplayFPSTimer = glfwGetTime();
	}

	void RenderSettingsPanel::UpdateMetrics(float deltaTime)
	{
		std::shared_ptr<Camera> camera = m_Context->GetScene()->GetCamera();

		m_NAccumulatedFrame++;
		m_NumRaysProcessed += camera->GetResolution().x * camera->GetResolution().y;

		m_AccumulatedTime += deltaTime;
		if (glfwGetTime() - m_DisplayFPSTimer >= 0.2f || m_DeltaTime == 0)
		{
			m_DisplayFPSTimer = glfwGetTime();
			m_DeltaTime = m_AccumulatedTime / m_NAccumulatedFrame;
			m_MRPS = static_cast<float>(m_NumRaysProcessed) / m_AccumulatedTime / 1000.0f;		// millisecond * 1.000.000

			m_NAccumulatedFrame = 0;
			m_AccumulatedTime = 0.0f;
			m_NumRaysProcessed = 0;
		}
	}

	void RenderSettingsPanel::OnImGuiRender(uint32_t frameNumber)
	{
		std::shared_ptr<Camera> camera = m_Context->GetScene()->GetCamera();

		ImGui::Begin("Render Settings");

		if (UI::BeginSection("Statistics"))
		{
			if (UI::BeginPropertyTable("statistics"))
			{
				UI::PropertyLabel("Render time");
				ImGui::Text("%.3f ms", m_DeltaTime);

				UI::PropertyLabel("FPS");
				ImGui::Text("%d", (int)(1000.0f / m_DeltaTime));

				UI::PropertyLabel("Frame");
				ImGui::Text("%d", frameNumber);

				UI::PropertyLabel("Rays/sec", "Megarays traced per second");
				ImGui::Text("%.2f M", m_MRPS);

				UI::EndPropertyTable();
			}

			UI::EndSection();
		}

		// TODO: move camera settings to another panel
		if (UI::BeginSection("Camera"))
		{
			if (UI::BeginPropertyTable("camera"))
			{
				// Labels lean on the section heading for context: "Horizontal FOV" only fits a wide
				// panel, and "FOV" under a "Camera" heading says the same thing.
				UI::PropertyLabel("FOV", "Horizontal field of view, in degrees");
				if (ImGui::SliderFloat("##fov", &camera->GetHorizontalFOV(), 1.0f, 180.0f))
					camera->MarkChanged();

				UI::PropertyLabel("Focus distance");
				if (ImGui::DragFloat("##focus", &camera->GetFocusDist(), 0.02f, 0.01f, 1000.0f))
					camera->MarkChanged();

				UI::PropertyLabel("Defocus angle");
				if (ImGui::DragFloat("##defocus", &camera->GetDefocusAngle(), 0.2f, 0.0f, 180.0f))
					camera->MarkChanged();

				UI::EndPropertyTable();
			}

			UI::EndSection();
		}

		RenderSettings& renderSettings = m_Context->GetScene()->GetRenderSettings();
		if (UI::BeginSection("Resolution"))
		{
			if (UI::BeginPropertyTable("resolution"))
			{
				ImGui::BeginDisabled(m_FitRenderToViewport);
				float2 resolution = make_float2(renderSettings.resolution);
				UI::PropertyLabel("Size");
				if (ImGui::InputFloat2("##resolution", (float*)&resolution, "%.0fpx"))
				{
					if (resolution.x > 0 && resolution.x <= MaxRenderResolution && resolution.y > 0 && resolution.y <= MaxRenderResolution)
					{
						m_Context->OnResize(make_uint2(resolution.x, resolution.y));
						m_Context->GetScene()->InvalidateAccumulation();
					}
				}
				ImGui::EndDisabled();

				UI::PropertyLabel("Match viewport", "Resize the render to the viewport panel");
				ImGui::Checkbox("##matchviewport", &m_FitRenderToViewport);

				UI::EndPropertyTable();
			}

			UI::EndSection();
		}

		if (UI::BeginSection("Render settings"))
		{
			if (UI::BeginPropertyTable("rendersettings"))
			{
				UI::PropertyLabel("MIS", "Multiple importance sampling");
				if (ImGui::Checkbox("##mis", &renderSettings.useMIS))
					m_Context->GetScene()->InvalidateAccumulation();

				UI::PropertyLabel("Visualize BVH");
				if (ImGui::Checkbox("##visualizebvh", &renderSettings.visualizeBvh))
					m_Context->GetScene()->InvalidateAccumulation();

				if (renderSettings.visualizeBvh)
				{
					UI::PropertyLabel("Wireframe");
					if (ImGui::Checkbox("##wireframe", &renderSettings.wireframeBvh))
						m_Context->GetScene()->InvalidateAccumulation();
				}

				int pathLength = renderSettings.pathLength;
				UI::PropertyLabel("Path length");
				if (ImGui::SliderInt("##pathlength", &pathLength, 1, PATH_MAX_LENGTH))
					m_Context->GetScene()->InvalidateAccumulation();
				renderSettings.pathLength = pathLength;

				UI::PropertyLabel("Background");
				if (ImGui::ColorEdit3("##backgroundcolor", (float*)&renderSettings.backgroundColor, ImGuiColorEditFlags_NoInputs))
					m_Context->GetScene()->InvalidateAccumulation();

				UI::PropertyLabel("Intensity", "Background intensity");
				if (ImGui::DragFloat("##backgroundintensity", &renderSettings.backgroundIntensity, 0.01f, 0.0f, 1000.0f))
					m_Context->GetScene()->InvalidateAccumulation();

				UI::EndPropertyTable();
			}

			UI::EndSection();
		}

		if (UI::BeginSection("Color management"))
		{
			if (UI::BeginPropertyTable("colormanagement"))
			{
				UI::PropertyLabel("Exposure");
				ImGui::DragFloat("##exposure", &renderSettings.exposure, 0.01f, 2.0f, 2.0f);

				UI::PropertyLabel("Tone mapping");
				int currentIndex = static_cast<int>(renderSettings.toneMapping);
				if (ImGui::Combo("##tonemapping", &currentIndex, ColorUtils::ToneMappingNames, IM_ARRAYSIZE(ColorUtils::ToneMappingNames)))
					renderSettings.toneMapping = static_cast<ColorUtils::ToneMapping>(currentIndex);

				UI::EndPropertyTable();
			}

			UI::EndSection();
		}

		ImGui::End();
	}

}
