#include "RenderPanel.h"

RenderPanel::RenderPanel(Renderer* renderer)
	: m_Renderer(renderer)
{
}

void RenderPanel::OnImGuiRender(bool fitRenderToViewport)
{
	const float maxZoom = std::pow(1.4f, 12.0f);
	const float minZoom = std::pow(1.4f, -2.0f);

	ImGui::Begin("Render", nullptr, ImGuiWindowFlags_NoScrollbar);

	float zoomDelta = Input::GetScrollOffsetY();
	if (zoomDelta != 0.0f)
	{
		ImVec2 mousePos = ImGui::GetMousePos();
		ImVec2 localMouse = mousePos - m_TopLeft;

		// Image pixel under mouse BEFORE zoom
		ImVec2 imagePosBefore = localMouse / m_RenderScale;

		// Apply zoom
		float newScale = clamp(m_RenderScale * std::pow(1.4f, zoomDelta), minZoom, maxZoom);

		// Pixel under mouse after zoom
		ImVec2 imagePosAfter = imagePosBefore * newScale;

		// New scroll so mouse stays on same pixel
		m_RenderScroll += imagePosAfter - localMouse;
		m_RenderScale = newScale;

		ImVec2 childSize = ImVec2(m_Renderer->GetResolution().x, m_Renderer->GetResolution().y) * m_RenderScale;
		//ImVec2 childSize = m_ViewportSize * m_RenderScale;
		childSize.x = std::max(m_ViewportSize.x, childSize.x);
		childSize.y = std::max(m_ViewportSize.y, childSize.y);
		ImGui::SetNextWindowContentSize(childSize);
		ImGui::SetNextWindowScroll(m_RenderScroll);
	}

	ImVec2 renderZoneSize = ImGui::GetContentRegionAvail();
	// Subtract the height of the second child
	renderZoneSize.y -= ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y;
	ImGui::BeginChild("Render zone", renderZoneSize, 0, ImGuiWindowFlags_AlwaysVerticalScrollbar | ImGuiWindowFlags_AlwaysHorizontalScrollbar);

	m_TopLeft = ImGui::GetCursorScreenPos();
	m_RenderScroll = ImVec2(ImGui::GetScrollX(), ImGui::GetScrollY());

	ImVec2 renderSize = ImVec2(m_Renderer->GetTexture().GetWidth(), m_Renderer->GetTexture().GetHeight()) * m_RenderScale;
	ImVec2 childSize = ImVec2(m_Renderer->GetResolution().x, m_Renderer->GetResolution().y) * m_RenderScale;
	childSize.x = std::max(m_ViewportSize.x, childSize.x);
	childSize.y = std::max(m_ViewportSize.y, childSize.y);

	// Very ugly workaround: when calling SetNextWindowContentSize,
	// the content region available is also modified for this frame and we don't want to account for that change
	if (zoomDelta == 0.0f)
		m_ViewportSize = ImGui::GetContentRegionAvail();

	if (fitRenderToViewport)
		m_Renderer->OnResize(make_uint2(m_ViewportSize.x, m_ViewportSize.y));

	ImGui::BeginChild("Render", childSize);

	// For debugging purposes
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
	ImGui::EndChild();

	ImGui::SetNextItemWidth(150 * ImGui::GetWindowDpiScale());

	int32_t zoomPercent = std::round(m_RenderScale * 100.0f);
	if (ImGui::SliderInt("Zoom", &zoomPercent, minZoom * 100.0f, maxZoom * 100.0f, "%d%%"))
		m_RenderScale = zoomPercent / 100.0f;

	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x * 2.0f, ImGui::GetStyle().FramePadding.y));
	ImVec2 buttonsSize = ImGui::CalcTextSize("Render") + ImGui::CalcTextSize("Export") + ImGui::GetStyle().FramePadding * 4.0f + ImGui::GetStyle().ItemSpacing * 2.0f;
	//ImVec2 buttonsSize = ImGui::CalcTextSize("Render") + ImGui::GetStyle().FramePadding * 4.0f;// +ImGui::GetStyle().ItemSpacing * 2.0f;

	ImGui::SameLine(m_ViewportSize.x - buttonsSize.x);
	if (ImGui::Button("Render") || ImGui::IsKeyPressed(ImGuiKey_F5))
		m_MustRender = !m_MustRender;

	ImGui::SameLine();
	if (ImGui::Button("Export"))
		m_ExportRender = true;

	ImGui::PopStyleVar();

	ImGui::End();
}

bool RenderPanel::ExportRender()
{
	bool exportRender = m_ExportRender;
	m_ExportRender = false;
	return exportRender;
}
