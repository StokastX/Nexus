#pragma once
#include "Scene/Scene.h"

class SceneHierarchyPanel
{
public:
	SceneHierarchyPanel(Scene* context);

	void SetContext(Scene* context);
	void SetSelectionContext(SelectionContext::Type type, int32_t idx);
	SelectionContext& GetSelectionContext() { return m_SelectionContext; }

	void OnImGuiRender();

private:
	void DrawProperties(SelectionContext selectionContext);

private:
	Scene* m_Context;
	SelectionContext m_SelectionContext = {SelectionContext::Type::INSTANCE, -1};
};
