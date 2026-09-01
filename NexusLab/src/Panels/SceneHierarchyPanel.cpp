#include "SceneHierarchyPanel.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"


namespace Nexus {

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
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("Hierarchy panel");
		ImGui::PopStyleVar();

		MirroredVector<MeshInstance>& meshInstances = m_Context->GetMeshInstances();
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
		for (int i = 0; i < meshInstances.Size(); i++)
		{
			const MeshInstance& meshInstance = meshInstances[i];
			bool itemSelected = m_SelectionContext.type == SelectionContext::Type::INSTANCE && m_SelectionContext.idx == i;
			ImGuiTreeNodeFlags flags = (itemSelected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow;
			flags |= ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowItemOverlap;
			bool opened = ImGui::TreeNodeEx(std::to_string(i).c_str(), flags, "%s", meshInstance.name.c_str());

			if (ImGui::IsItemClicked())
				SetSelectionContext(SelectionContext::Type::INSTANCE, i);

			if (opened)
				ImGui::TreePop();
		}

		MirroredVector<Light>& lights = m_Context->GetLights();
		for (uint32_t i = 0; i < lights.Size(); i++)
		{
			const Light& light = lights[i];
			if (light.type == Light::Type::MESH)
				continue;

			bool itemSelected = m_SelectionContext.type == SelectionContext::Type::LIGHT && m_SelectionContext.idx == i;
			ImGuiTreeNodeFlags flags = (itemSelected ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow;
			flags |= ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowItemOverlap;
			bool opened = ImGui::TreeNodeEx(std::to_string(i).c_str(), flags, "Light %u", i);

			if (ImGui::IsItemClicked())
				SetSelectionContext(SelectionContext::Type::LIGHT, i);

			if (opened)
				ImGui::TreePop();
		}
		ImGui::PopStyleVar();

		if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsWindowHovered())
			m_SelectionContext.idx = -1;

		ImGui::End();

		ImGui::Begin("Properties");
		if (m_SelectionContext.idx != -1)
			DrawProperties(m_SelectionContext);

		ImGui::End();
	}

	static bool DrawFloat3Control(const std::string& label, float3& values, float resetValue = 0.0f, float step = 0.1f, const char* format = "%.2f", float columnWidth = 50.0f)
	{
		ImGui::PushID(label.c_str());

		ImGui::Columns(2);
		ImGui::SetColumnWidth(0, columnWidth * ImGui::GetWindowDpiScale());
		ImGui::Text("%s", label.c_str());
		ImGui::NextColumn();

		ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImGui::GetStyle().ItemSpacing * 0.5f);

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

		ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 0.0f);
		if (selectionContext.type == SelectionContext::Type::INSTANCE)
		{
			auto meshInstance = m_Context->GetMeshInstances().Mutate(selectionContext.idx);

			if (ImGui::TreeNodeEx("Transform", flags))
			{
				DrawFloat3Control("Location", meshInstance->position);
				DrawFloat3Control("Rotation", meshInstance->rotation);
				DrawFloat3Control("Scale", meshInstance->scale, 1.0f, 0.01f, "%.3f");

				ImGui::TreePop();
			}

			AssetManager& assetManager = m_Context->GetAssetManager();

			MirroredVector<Material>& materials = assetManager.GetMaterials();
			std::string materialsString = assetManager.GetMaterialsString();

			if (ImGui::TreeNodeEx("Material", flags))
			{
				if (meshInstance->materialIdx == -1)
				{
					if (ImGui::Button("Custom material"))
						meshInstance->materialIdx = 0;
				}
				else
				{
					int materialIdx = meshInstance->materialIdx;
					if (ImGui::Combo("Id", &materialIdx, materialsString.c_str()))
						meshInstance->materialIdx = materialIdx;

					auto material = materials.Mutate(meshInstance->materialIdx);

					ImGui::ColorEdit3("Base color", (float*)&material->baseColor);
					ImGui::DragFloat("Metalness", &material->metalness, 0.01f, 0.0f, 1.0f);
					ImGui::DragFloat("Roughness", &material->roughness, 0.01f, 0.0f, 1.0f);
					ImGui::DragFloat("IOR", &material->ior, 0.01f, 1.0f, 2.5f);
					ImGui::DragFloat("Transmission", &material->transmission, 0.01f, 0.0f, 1.0f);
					ImGui::DragFloat("Opacity", (float*)&material->opacity, 0.01f, 0.0f, 1.0f);
					if (ImGui::TreeNodeEx("Specular", flags))
					{
						ImGui::DragFloat("Specular weight", &material->specularWeight, 0.01f, 0.0f, 1.0f);
						ImGui::ColorEdit3("Specular color", (float*)&material->specularColor);
						ImGui::DragFloat("Anisotropy", &material->anisotropy, 0.01f, 0.0f, 1.0f);
						ImGui::TreePop();
					}
					if (ImGui::TreeNodeEx("Emission", flags & ~ImGuiTreeNodeFlags_DefaultOpen))
					{
						ImGui::ColorEdit3("Emission color", (float*)&material->emissionColor);
						ImGui::DragFloat("Intensity", (float*)&material->intensity, 0.1f, 0.0f, 1000.0f);
						ImGui::TreePop();
					}
				}
				ImGui::TreePop();
			}
		}
		else if (selectionContext.type == SelectionContext::Type::LIGHT)
		{
			auto light = m_Context->GetLights().Mutate(selectionContext.idx);
			if (ImGui::TreeNodeEx("Light", flags))
			{
				int currentIndex = static_cast<int>(light->type);
				if (ImGui::Combo("Type", &currentIndex, lightTypeNames, IM_ARRAYSIZE(lightTypeNames)))
					light->type = static_cast<Light::Type>(currentIndex);

				switch (light->type)
				{
				case Light::Type::POINT:
					DrawFloat3Control("Location", light->point.position);
					ImGui::ColorEdit3("Color", (float*)&light->point.color);
					ImGui::DragFloat("Intensity", &light->point.intensity, 0.1f, 0.0f, 1000.0f);
					break;

				case Light::Type::SPOT:
					DrawFloat3Control("Location", light->spot.position);
					ImGui::ColorEdit3("Color", (float*)&light->spot.color);
					ImGui::DragFloat("Intensity", &light->spot.intensity, 0.1f, 0.0f, 1000.0f);
					ImGui::DragFloat("Falloff Start", &light->spot.falloffStart, 0.1f, 0.0f, 180.0f);
					ImGui::DragFloat("Falloff End", &light->spot.falloffEnd, 0.1f, 0.0f, 180.0f);
					break;

				case Light::Type::DIRECTIONAL:
					DrawFloat3Control("Direction", light->directional.direction);
					ImGui::ColorEdit3("Color", (float*)&light->directional.color);
					ImGui::DragFloat("Intensity", &light->directional.intensity, 0.1f, 0.0f, 1000.0f);
					break;
				default:
					break;
				}
				ImGui::TreePop();
			}
		}
		ImGui::PopStyleVar();
	}

}
