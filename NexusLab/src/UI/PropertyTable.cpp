#include "PropertyTable.h"

#include <cfloat>
#include <cstdio>

namespace Nexus::UI {

	bool BeginPropertyTable(const char* id, float labelWeight)
	{
		const ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_NoBordersInBody
			| ImGuiTableFlags_SizingStretchProp;

		// A table row is padded by CellPadding.y top and bottom, so the panel default adds its full
		// height to every property. Rows are already separated by the gap between framed widgets;
		// this keeps the list as tight as the plain widget list it replaced. Kept as a fraction of
		// the style value so it still tracks the DPI scaling applied in ImGuiLayer::OnAttach.
		const ImVec2 cellPadding = ImGui::GetStyle().CellPadding;
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(cellPadding.x, cellPadding.y * 0.25f));

		if (!ImGui::BeginTable(id, 2, flags))
		{
			ImGui::PopStyleVar();
			return false;
		}

		ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch, labelWeight);
		ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 1.0f - labelWeight);
		return true;
	}

	void EndPropertyTable()
	{
		ImGui::EndTable();
		ImGui::PopStyleVar();
	}

	void PropertyLabel(const char* label, const char* tooltip)
	{
		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::AlignTextToFramePadding();

		const float available = ImGui::GetContentRegionAvail().x;
		const float textWidth = ImGui::CalcTextSize(label).x;

		// Right-aligned, so every label ends against the gutter and sits next to the widget it
		// names. Label lengths here range from "MIS" to "Match viewport", and left-aligning strands
		// the short ones a long way from their field.
		//
		// Clamped at the cell start rather than allowed to go negative: a label wider than the
		// column would otherwise be pushed left out of the cell and clipped at its *beginning*,
		// hiding the part that identifies it. Overlong labels fall back to left-aligned and clip at
		// the end, where the tooltip picks them up.
		if (textWidth < available)
			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + available - textWidth);

		ImGui::TextUnformatted(label);
		if (ImGui::IsItemHovered() && (tooltip != nullptr || textWidth > available))
			ImGui::SetTooltip("%s", tooltip != nullptr ? tooltip : label);

		ImGui::TableSetColumnIndex(1);
		ImGui::SetNextItemWidth(-FLT_MIN);
	}

	bool DragFloat3Row(const char* label, float3& values, float step, const char* format,
		const char* tooltip)
	{
		ImGui::PushID(label);

		float* component[3] = { &values.x, &values.y, &values.z };
		const char* axisLabel[3] = { "X", "Y", "Z" };
		const char* fieldLabel[3] = { "##X", "##Y", "##Z" };

		bool modified = false;

		// One row per axis, the way Blender stacks vector properties. Three fields side by side left
		// each about 55px at a typical panel width, which truncated "1.000" to "1.0(".
		//
		// The axis letter is simply the end of the label text. Because labels are right-aligned, the
		// letter lands against the gutter on every row and the three line up under each other with
		// no positioning of their own.
		for (int axis = 0; axis < 3; axis++)
		{
			char rowLabel[128];
			if (axis == 0)
				snprintf(rowLabel, sizeof(rowLabel), "%s %s", label, axisLabel[axis]);
			else
				snprintf(rowLabel, sizeof(rowLabel), "%s", axisLabel[axis]);

			PropertyLabel(rowLabel, axis == 0 ? tooltip : nullptr);

			if (ImGui::DragFloat(fieldLabel[axis], component[axis], step, 0.0f, 0.0f, format))
				modified = true;
		}

		ImGui::PopID();

		return modified;
	}

}
