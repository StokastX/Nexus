#include "ViewportPanel.h"

ViewportPanel::ViewportPanel(Renderer* renderer)
	: m_Renderer(renderer)
{
}

void ViewportPanel::OnImGuiRender(bool fitRenderToViewport)
{
	float zoomDelta = 0.1f * Input::GetScrollOffsetY();
	if (!fitRenderToViewport)
	{
		if (zoomDelta != 0.0f)
		{
			ImVec2 mousePos = ImGui::GetMousePos();
			ImVec2 localMouse = mousePos - m_TopLeft;

			// Image pixel under mouse BEFORE zoom
			ImVec2 imagePosBefore = localMouse / m_RenderScale;

			// Apply zoom
			float newScale = clamp(m_RenderScale * std::pow(4.0f, zoomDelta), 0.1f, 10.0f);

			// Pixel under mouse after zoom
			ImVec2 imagePosAfter = imagePosBefore * newScale;

			m_RenderScroll += imagePosAfter - localMouse;
			// New scroll so mouse stays on same pixel
			ImGui::SetNextWindowScroll(m_RenderScroll);

			m_RenderScale = newScale;
		}
	}

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_AlwaysHorizontalScrollbar);

	if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !m_Renderer->GetScene()->IsEmpty())
	{
		ImVec2 viewportPos = ImGui::GetCursorScreenPos();
		ImVec2 mousePos = ImGui::GetMousePos();
		int2 hoveredPixel = make_int2(mousePos.x - viewportPos.x, mousePos.y - viewportPos.y);
		uint2 resolution = m_Renderer->GetTexture().GetResolution();
		hoveredPixel.y = resolution.y - hoveredPixel.y;

		if (hoveredPixel.x >= 0 && hoveredPixel.x < resolution.x && hoveredPixel.y >= 0 && hoveredPixel.y < resolution.y)
		{
			m_Renderer->GetPathTracer()->SetPixelQuery(hoveredPixel.x, hoveredPixel.y);
		}
	}

	m_TopLeft = ImGui::GetCursorScreenPos();
	m_RenderScroll = ImVec2(ImGui::GetScrollX(), ImGui::GetScrollY());

	ImVec2 renderSize, childSize;
	if (!fitRenderToViewport)
	{
		renderSize = ImVec2(m_Renderer->GetTexture().GetWidth(), m_Renderer->GetTexture().GetHeight()) * m_RenderScale;
		childSize = ImGui::GetContentRegionAvail() * m_RenderScale;
		childSize.x = std::max(ImGui::GetContentRegionAvail().x, childSize.x);
		childSize.y = std::max(ImGui::GetContentRegionAvail().y, childSize.y);
	}
	else
	{
		ImVec2 viewportSize = ImGui::GetContentRegionAvail();
		if (viewportSize != m_ViewportSize)
			m_Renderer->OnResize(make_uint2(viewportSize.x, viewportSize.y));
		childSize = viewportSize;
		renderSize = viewportSize;
	}

	ImGui::BeginChild("Render", childSize);

	ImGui::SetCursorPos(ImGui::GetCursorPos() + (childSize - renderSize) * 0.5f);

	ImGui::Image((void*)(intptr_t)m_Renderer->GetTexture().GetHandle(), renderSize, ImVec2(0, 1), ImVec2(1, 0));

	ImGui::EndChild();


	ImGui::End();
	ImGui::PopStyleVar();
}
