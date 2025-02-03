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

void SceneHierarchyPanel::SetSelectionContext(int selectionContext)
{
	m_SelectionContext = selectionContext;
}

void SceneHierarchyPanel::OnImGuiRender()
{
	ImGui::Begin("Hierarchy panel");

	std::vector<MeshInstance>& meshInstances =  m_Context->GetMeshInstances();
	for (int i = 0; i < meshInstances.size(); i++)
	{
		MeshInstance& meshInstance = meshInstances[i];
		ImGuiTreeNodeFlags flags = (m_SelectionContext == i ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow;
		flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
		bool opened = ImGui::TreeNodeEx((void*)i, flags, meshInstance.name.c_str());

		if (ImGui::IsItemClicked())
		{
			m_SelectionContext = i;
		}

		if (opened)
			ImGui::TreePop();
	}

	if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsWindowHovered())
		m_SelectionContext = -1;

	ImGui::End();

	ImGui::Begin("Properties");
	if (m_SelectionContext != -1)
		DrawProperties(m_SelectionContext);

	ImGui::End();
}

static bool DrawFloat3Control(const std::string& label, float3& values, float resetValue = 0.0f, float step = 0.1f, const char* format = "%.2f", float columnWidth = 80.0f)
{
	ImGui::PushID(label.c_str());

	ImGui::Columns(2);
	ImGui::SetColumnWidth(0, columnWidth);
	ImGui::Text(label.c_str());
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
		values.y = resetValue, modified = true;;
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

void SceneHierarchyPanel::DrawProperties(int selectionContext)
{
	MeshInstance& meshInstance = m_Context->GetMeshInstances()[selectionContext];
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_Framed
		| ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_SpanAvailWidth;

	if (ImGui::TreeNodeEx("Transform", flags))
	{
		if (DrawFloat3Control("Location", meshInstance.position))
			m_Context->InvalidateMeshInstance(selectionContext);

		if (DrawFloat3Control("Rotation", meshInstance.rotation))
			m_Context->InvalidateMeshInstance(selectionContext);

		if (DrawFloat3Control("Scale", meshInstance.scale, 1.0f, 0.01f, "%.3f"))
			m_Context->InvalidateMeshInstance(selectionContext);

		ImGui::TreePop();
	}

	AssetManager& assetManager = m_Context->GetAssetManager();

	std::vector<Material>& materials = assetManager.GetMaterials();
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
			if (ImGui::Combo("Id", &meshInstance.materialIdx, materialsString.c_str()))
				m_Context->InvalidateMeshInstance(selectionContext);

			Material& material = materials[meshInstance.materialIdx];
			int type = (int)material.type;

			if (ImGui::Combo("Type", &type, materialTypes.c_str()))
				assetManager.InvalidateMaterial(meshInstance.materialIdx);

			material.type = (Material::Type)type;

			switch (material.type)
			{
			case Material::Type::DIFFUSE:
				if (ImGui::ColorEdit3("Albedo", (float*)&material.diffuse.albedo))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				break;
			case Material::Type::DIELECTRIC:
				if (ImGui::ColorEdit3("Albedo", (float*)&material.dielectric.albedo))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat("Roughness", &material.dielectric.roughness, 0.01f, 0.0f, 1.0f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat("Refraction index", &material.dielectric.ior, 0.01f, 1.0f, 2.5f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				break;
			case Material::Type::PLASTIC:
				if (ImGui::ColorEdit3("Albedo", (float*)&material.plastic.albedo))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat("Roughness", &material.plastic.roughness, 0.01f, 0.0f, 1.0f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat("Refraction index", &material.plastic.ior, 0.01f, 1.0f, 2.5f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				break;
			case Material::Type::CONDUCTOR:
				if (ImGui::DragFloat("Roughness", &material.conductor.roughness, 0.01f, 0.0f, 1.0f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat3("Refraction index", (float*)&material.conductor.ior, 0.01f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				if (ImGui::DragFloat3("k", (float*)&material.conductor.k, 0.01f))
					assetManager.InvalidateMaterial(meshInstance.materialIdx);
				break;
			}
			if (ImGui::ColorEdit3("Emission", (float*)&material.emissive))
			{
				// Invalidate mesh instance to update lighting
				m_Context->InvalidateMeshInstance(selectionContext);
				assetManager.InvalidateMaterial(meshInstance.materialIdx);
			}
			if (ImGui::DragFloat("Intensity", (float*)&material.intensity, 0.1f, 0.0f, 1000.0f))
			{
				m_Context->InvalidateMeshInstance(selectionContext);
				assetManager.InvalidateMaterial(meshInstance.materialIdx);
			}
			if (ImGui::DragFloat("Opacity", (float*)&material.opacity, 0.01f, 0.0f, 1.0f))
			{
				assetManager.InvalidateMaterial(meshInstance.materialIdx);
			}
		}
		ImGui::TreePop();
	}
}
