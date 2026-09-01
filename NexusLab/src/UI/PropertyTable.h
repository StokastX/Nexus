#pragma once

#include <imgui.h>
#include "Utils/cuda_math.h"

namespace Nexus::UI {

	/*
	 * Two-column property layout: labels on the left, widgets on the right, every value widget
	 * sharing one edge regardless of how long the labels are.
	 *
	 * The split is a fraction of the panel rather than a pixel count, so it holds its proportions
	 * when the panel is resized, and the column is draggable and persisted in imgui.ini per table
	 * id. Labels are still the first thing to lose room when a panel gets narrow -- that is
	 * unavoidable, so PropertyLabel clips them and offers the full text as a tooltip rather than
	 * pushing the value column around.
	 */

	// labelWeight is the share of the panel the label column takes. Every table in a panel should
	// normally share one value so their value columns line up; lower it only where a section is
	// made entirely of short labels.
	bool BeginPropertyTable(const char* id, float labelWeight = 0.4f);
	void EndPropertyTable();

	// Emits the label cell and leaves the cursor in the value cell with the item width already
	// stretched to fill it. Follow it with exactly one widget, labelled "##something" so the
	// widget does not draw a second copy of the label.
	void PropertyLabel(const char* label, const char* tooltip = nullptr);

	// A vector property: three axis-coloured drag fields stacked one per row, the way Blender lays
	// out vectors, with the label on the first row only. Returns true on any edit.
	bool DragFloat3Row(const char* label, float3& values, float step = 0.1f,
		const char* format = "%.2f", const char* tooltip = nullptr);

}
