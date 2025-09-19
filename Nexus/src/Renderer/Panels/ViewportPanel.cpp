#include "ViewportPanel.h"

ViewportPanel::ViewportPanel(OGLRenderer* renderer)
	: m_Renderer(renderer)
{
}

void ViewportPanel::OnImGuiRender()
{
	ImGui::Begin("Viewport");
	//if (ImGui::IsActive)

	if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
	{
		ImVec2 viewportPos = ImGui::GetCursorScreenPos();
		ImVec2 mousePos = ImGui::GetMousePos();
		float2 hoveredPixel = make_float2(mousePos.x - viewportPos.x, mousePos.y - viewportPos.y);
		uint2 resolution = m_Renderer->GetTexture().GetResolution();
		hoveredPixel.y = resolution.y - hoveredPixel.y;

		if (hoveredPixel.x >= 0 && hoveredPixel.x < resolution.x && hoveredPixel.y >= 0 && hoveredPixel.y < resolution.y)
		{
			m_Renderer->SetPixelQuery(make_uint2(hoveredPixel.x, hoveredPixel.y));
		}
	}
	ImVec2 renderSize = ImGui::GetContentRegionAvail();
	m_Renderer->OnResize(make_uint2(renderSize.x, renderSize.y));

	ImGui::Image((ImTextureID)m_Renderer->GetTexture().GetHandle(), renderSize, ImVec2(0, 1), ImVec2(1, 0));

	ImGui::End();
}
