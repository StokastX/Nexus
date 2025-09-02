#include "ViewportPanel.h"

ViewportPanel::ViewportPanel(Renderer* renderer)
	: m_Renderer(renderer)
{
}

void ViewportPanel::OnImGuiRender(bool fitRenderToViewport)
{
	float zoomDelta = Input::GetScrollOffsetY();
	if (!fitRenderToViewport)
	{
	  if (zoomDelta != 0.0f)
	  {
	  	ImVec2 mousePos = ImGui::GetMousePos();
	  	ImVec2 localMouse = mousePos - m_TopLeft;

	  	// Image pixel under mouse BEFORE zoom
	  	ImVec2 imagePosBefore = localMouse / m_RenderScale;

	  	// Apply zoom
		const float maxZoom = std::pow(1.4f, 12.0f);
		const float minZoom = std::pow(1.4f, -2.0f);
	  	float newScale = clamp(m_RenderScale * std::pow(1.4f, zoomDelta), minZoom, maxZoom);

	  	// Pixel under mouse after zoom
	  	ImVec2 imagePosAfter = imagePosBefore * newScale;

	  	// New scroll so mouse stays on same pixel
	  	m_RenderScroll += imagePosAfter - localMouse;
	  	m_RenderScale = newScale;

		ImVec2 childSize = m_ViewportSize * m_RenderScale;
		childSize.x = std::max(m_ViewportSize.x, childSize.x);
		childSize.y = std::max(m_ViewportSize.y, childSize.y);
		ImGui::SetNextWindowContentSize(childSize);
	  	ImGui::SetNextWindowScroll(m_RenderScroll);
	  }
	}

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_AlwaysHorizontalScrollbar);

	m_TopLeft = ImGui::GetCursorScreenPos();
	m_RenderScroll = ImVec2(ImGui::GetScrollX(), ImGui::GetScrollY());

	ImVec2 renderSize, childSize;
	if (!fitRenderToViewport)
	{
		renderSize = ImVec2(m_Renderer->GetTexture().GetWidth(), m_Renderer->GetTexture().GetHeight()) * m_RenderScale;
		childSize = m_ViewportSize * m_RenderScale;
		childSize.x = std::max(m_ViewportSize.x, childSize.x);
		childSize.y = std::max(m_ViewportSize.y, childSize.y);
	}
	else
	{
		m_RenderScale = 1.0f;
		ImVec2 viewportSize = ImGui::GetContentRegionAvail();
		m_Renderer->OnResize(make_uint2(viewportSize.x, viewportSize.y));
		childSize = viewportSize;
		renderSize = viewportSize;
	}
	// Very ugly workaround: when calling SetNextWindowContentSize,
	// the content region available is also modified for this frame and we don't want to account for that change
	if (zoomDelta == 0.0f)
		m_ViewportSize = ImGui::GetContentRegionAvail();

	ImGui::BeginChild("Render", childSize);

	if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !m_Renderer->GetScene()->IsEmpty())
	{
		ImVec2 viewportPos = ImGui::GetCursorScreenPos();
		ImVec2 mousePos = ImGui::GetMousePos();
		float2 hoveredPixel = make_float2(mousePos.x - viewportPos.x, mousePos.y - viewportPos.y) / m_RenderScale;
		uint2 resolution = m_Renderer->GetResolution();
		hoveredPixel.y = resolution.y - hoveredPixel.y;

		if (hoveredPixel.x >= 0 && hoveredPixel.x < resolution.x && hoveredPixel.y >= 0 && hoveredPixel.y < resolution.y)
		{
			m_Renderer->GetPathTracer()->SetPixelQuery(hoveredPixel.x, hoveredPixel.y);
		}
	}

	ImGui::SetCursorPos(ImGui::GetCursorPos() + (childSize - renderSize) * 0.5f);

	ImGui::Image((ImTextureID)m_Renderer->GetTexture().GetHandle(), renderSize, ImVec2(0, 1), ImVec2(1, 0));

	ImGui::EndChild();


	ImGui::End();
	ImGui::PopStyleVar();
}
