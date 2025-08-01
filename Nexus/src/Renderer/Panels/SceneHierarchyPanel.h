#pragma once
#include "Scene/Scene.h"

struct SelectionContext
{
	enum struct Type
	{
		INSTANCE,
		LIGHT
	};
	Type type;
	int32_t idx;
};

class SceneHierarchyPanel
{
public:
	SceneHierarchyPanel(Scene* context);

	void SetContext(Scene* context);
	void SetSelectionContext(SelectionContext::Type type, int32_t idx);

	void OnImGuiRender();

private:
	void DrawProperties(SelectionContext selectionContext);

private:
	Scene* m_Context;
	SelectionContext m_SelectionContext = {SelectionContext::Type::INSTANCE, -1};
};
