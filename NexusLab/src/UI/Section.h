#pragma once

namespace Nexus::UI {

	/*
	 * A collapsible panel section: a semibold header bar with its contents indented under it.
	 *
	 * Open/closed state is keyed on the label and persisted in imgui.ini, so a section the user
	 * folds stays folded across runs.
	 *
	 *     if (UI::BeginSection("Transform"))
	 *     {
	 *         ...
	 *         UI::EndSection();
	 *     }
	 *
	 * Call EndSection only when BeginSection returned true: a closed section indents nothing.
	 * Sections nest -- a section opened inside another is indented one further level, which is what
	 * makes Specular and Emission read as parts of Material rather than siblings of it.
	 */

	bool BeginSection(const char* label, bool defaultOpen = true);
	void EndSection();

}
