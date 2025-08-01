#include "SceneHierarchyPanel.h"
#include "imgui.h"
#include "imgui_internal.h"

SceneHierarchyPanel::SceneHierarchyPanel(Scene* context)
{
	SetContext(context);
}

void SceneHierarchyPanel::SetContext(Scene* context)
{
	m_Context = context;
}

void SceneHierarchyPanel::SetSelectionContext(SelectionContext::Type type, int32_t idx)
{
	m_SelectionContext.type = type;
	m_SelectionContext.idx = idx;
}

void SceneHierarchyPanel::OnImGuiRender()
{
	ImGui::Begin("Hierarchy panel");

	std::vector<MeshInstance>& meshInstances =  m_Context->GetMeshInstances();
	for (int i = 0; i < meshInstances.size(); i++)
	{
		MeshInstance& meshInstance = meshInstances[i];
		bool itemSelected = m_SelectionContext.type == SelectionContext::Type::INSTANCE && m_SelectionContext.idx == i;
		ImGuiTreeNodeFlags flags = (itemSelected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow;
		flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
		bool opened = ImGui::TreeNodeEx(std::to_string(i).c_str(), flags, "%s", meshInstance.name.c_str());

		if (ImGui::IsItemClicked())
			SetSelectionContext(SelectionContext::Type::INSTANCE, i);

		if (opened)
			ImGui::TreePop();
	}

	std::vector<Light>& lights = m_Context->GetLights();
	for (uint32_t i = 0; i < lights.size(); i++)
	{
		const Light& light = lights[i];
		if (light.type == Light::Type::MESH)
			continue;

		bool itemSelected = m_SelectionContext.type == SelectionContext::Type::LIGHT && m_SelectionContext.idx == i;
		ImGuiTreeNodeFlags flags = (itemSelected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow;
		flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
		bool opened = ImGui::TreeNodeEx(std::to_string(i).c_str(), flags, "Light %u", i);

		if (ImGui::IsItemClicked())
			SetSelectionContext(SelectionContext::Type::LIGHT, i);

		if (opened)
			ImGui::TreePop();
	}

	if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsWindowHovered())
		m_SelectionContext.idx = -1;

	ImGui::End();

	ImGui::Begin("Properties");
	if (m_SelectionContext.idx != -1)
		DrawProperties(m_SelectionContext);

	ImGui::End();
}

static bool DrawFloat3Control(const std::string& label, float3& values, float resetValue = 0.0f, float step = 0.1f, const char* format = "%.2f", float columnWidth = 80.0f)
{
	ImGui::PushID(label.c_str());

	ImGui::Columns(2);
	ImGui::SetColumnWidth(0, columnWidth);
	ImGui::Text("%s", label.c_str());
	ImGui::NextColumn();

	ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

	float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
	ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };

	bool modified = false;

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
	if (ImGui::Button("X", buttonSize))
		values.x = resetValue, modified = true;
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	if (ImGui::DragFloat("##X", &values.x, step, 0.0f, 0.0f, format))
		modified = true;
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
	if (ImGui::Button("Y", buttonSize))
		values.y = resetValue, modified = true;
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	if (ImGui::DragFloat("##Y", &values.y, step, 0.0f, 0.0f, format))
		modified = true;
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
	if (ImGui::Button("Z", buttonSize))
		values.z = resetValue, modified = true;
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	if (ImGui::DragFloat("##Z", &values.z, step, 0.0f, 0.0f, format))
		modified = true;
	ImGui::PopItemWidth();

	ImGui::PopStyleVar();

	ImGui::Columns(1);

	ImGui::PopID();

	return modified;
}

void SceneHierarchyPanel::DrawProperties(SelectionContext selectionContext)
{
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_Framed
		| ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_SpanAvailWidth;

	if (selectionContext.type == SelectionContext::Type::INSTANCE)
	{
		MeshInstance &meshInstance = m_Context->GetMeshInstances()[selectionContext.idx];

		if (ImGui::TreeNodeEx("Transform", flags))
		{
			if (DrawFloat3Control("Location", meshInstance.position))
				m_Context->InvalidateMeshInstance(selectionContext.idx);

			if (DrawFloat3Control("Rotation", meshInstance.rotation))
				m_Context->InvalidateMeshInstance(selectionContext.idx);

			if (DrawFloat3Control("Scale", meshInstance.scale, 1.0f, 0.01f, "%.3f"))
				m_Context->InvalidateMeshInstance(selectionContext.idx);

			ImGui::TreePop();
		}

		AssetManager &assetManager = m_Context->GetAssetManager();

		std::vector<Material> &materials = assetManager.GetMaterials();
		std::string materialsString = assetManager.GetMaterialsString();
		std::string materialTypes = Material::GetMaterialTypesString();

		if (ImGui::TreeNodeEx("Material", flags))
		{
			if (meshInstance.materialIdx == -1)
			{
				if (ImGui::Button("Custom material"))
					meshInstance.materialIdx = 0;
			}
			else
			{
				int materialIdx = meshInstance.materialIdx;
				// if (ImGui::Combo("Id", &materialIdx, materialsString.c_str()))
				//	m_Context->InvalidateMeshInstance(selectionContext);

				meshInstance.materialIdx = materialIdx;

				Material &material = materials[meshInstance.materialIdx];

				if (ImGui::ColorEdit3("Base color", (float *)&material.baseColor))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat("Metalness", &material.metalness, 0.01f, 0.0f, 1.0f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat("Roughness", &material.roughness, 0.01f, 0.0f, 1.0f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat("Specular weight", &material.specularWeight, 0.01f, 0.0f, 1.0f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::ColorEdit3("Specular color", (float *)&material.specularColor))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat("IOR", &material.ior, 0.01f, 1.0f, 2.5f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat("Transmission", &material.transmission, 0.01f, 0.0f, 1.0f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::ColorEdit3("Emission color", (float *)&material.emissionColor))
				{
					// Invalidate mesh instance to update lighting
					m_Context->InvalidateMeshInstance(selectionContext.idx);
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				}
				if (ImGui::DragFloat("Intensity", (float *)&material.intensity, 0.1f, 0.0f, 1000.0f))
				{
					m_Context->InvalidateMeshInstance(selectionContext.idx);
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				}
				if (ImGui::DragFloat("Opacity", (float *)&material.opacity, 0.01f, 0.0f, 1.0f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
			}
			ImGui::TreePop();
		}
	}
	else if (selectionContext.type == SelectionContext::Type::LIGHT)
	{
		Light& light = m_Context->GetLights()[selectionContext.idx];
		if (ImGui::TreeNodeEx("Light", flags))
		{
			int currentIndex = static_cast<int>(light.type);
			if (ImGui::Combo("Type", &currentIndex, lightTypeNames, IM_ARRAYSIZE(lightTypeNames)))
				light.type = static_cast<Light::Type>(currentIndex);

			switch (light.type)
			{
			case Light::Type::POINT:
				if (DrawFloat3Control("Location", light.point.position))
					m_Context->InvalidateLight(selectionContext.idx);
				if (ImGui::ColorEdit3("Color", (float*)&light.point.color))
					m_Context->InvalidateLight(selectionContext.idx);
				if (ImGui::DragFloat("Intensity", &light.point.intensity, 0.1f, 0.0f, 1000.0f))
					m_Context->InvalidateLight(selectionContext.idx);
				break;

			case Light::Type::SPOT:
				if (DrawFloat3Control("Location", light.spot.position))
					m_Context->InvalidateLight(selectionContext.idx);
				if (ImGui::ColorEdit3("Color", (float*)&light.spot.color))
					m_Context->InvalidateLight(selectionContext.idx);
				if (ImGui::DragFloat("Intensity", &light.spot.intensity, 0.1f, 0.0f, 1000.0f))
					m_Context->InvalidateLight(selectionContext.idx);
				if (ImGui::DragFloat("Falloff Start", &light.spot.falloffStart, 0.1f, 0.0f, 180.0f))
					m_Context->InvalidateLight(selectionContext.idx);
				if (ImGui::DragFloat("Falloff End", &light.spot.falloffEnd, 0.1f, 0.0f, 180.0f))
					m_Context->InvalidateLight(selectionContext.idx);
				break;

			case Light::Type::DIRECTIONAL:
				if (DrawFloat3Control("Direction", light.directional.direction))
					m_Context->InvalidateLight(selectionContext.idx);
				if (ImGui::ColorEdit3("Color", (float*)&light.directional.color))
					m_Context->InvalidateLight(selectionContext.idx);
				if (ImGui::DragFloat("Intensity", &light.directional.intensity, 0.1f, 0.0f, 1000.0f))
					m_Context->InvalidateLight(selectionContext.idx);
				break;
			default:
				break;
			}
			ImGui::TreePop();
		}
	}
}
