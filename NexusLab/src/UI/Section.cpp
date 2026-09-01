#include "Section.h"

#include <imgui.h>
#include "ImGui/ImGuiLayer.h"

namespace Nexus::UI {

	bool BeginSection(const char* label, bool defaultOpen)
	{
		ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding
			| ImGuiTreeNodeFlags_SpanAvailWidth;

		if (defaultOpen)
			flags |= ImGuiTreeNodeFlags_DefaultOpen;

		// Without this the gap above a header is the same as the gap between two rows, so the bar
		// reads as belonging to the section it follows rather than the one it opens.
		ImGui::Spacing();

		// Weight, not colour, is what separates a heading from its contents here: the header already
		// has a filled bar behind it, and dimming text on a lighter fill only costs contrast.
		ImFont* semiBold = ImGuiLayer::SemiBoldFont();
		if (semiBold != nullptr)
			ImGui::PushFont(semiBold);

		const bool open = ImGui::CollapsingHeader(label, flags);

		if (semiBold != nullptr)
			ImGui::PopFont();

		if (open)
			ImGui::Indent();

		return open;
	}

	void EndSection()
	{
		ImGui::Unindent();
	}

}
