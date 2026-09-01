#include "SceneHierarchyPanel.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"
#include "UI/PropertyTable.h"
#include "UI/Section.h"


namespace Nexus {

	// Outliner rows have no children yet. Leaf suppresses the disclosure arrow -- every row used
	// to draw one that did nothing -- and NoTreePushOnOpen means no matching TreePop. They stay
	// tree nodes rather than Selectables so transform parenting can nest under them later.
	static constexpr ImGuiTreeNodeFlags LeafFlags = ImGuiTreeNodeFlags_Leaf
		| ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanFullWidth
		| ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowItemOverlap;

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
		ImGui::Begin("Outliner");
		ImGui::PopStyleVar();

		MirroredVector<MeshInstance>& meshInstances = m_Context->GetMeshInstances();
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
		for (int i = 0; i < meshInstances.Size(); i++)
		{
			const MeshInstance& meshInstance = meshInstances[i];
			bool itemSelected = m_SelectionContext.type == SelectionContext::Type::INSTANCE && m_SelectionContext.idx == i;
			ImGuiTreeNodeFlags flags = (itemSelected ? ImGuiTreeNodeFlags_Selected : 0) | LeafFlags;
			ImGui::TreeNodeEx(std::to_string(i).c_str(), flags, "%s", meshInstance.name.c_str());

			if (ImGui::IsItemClicked())
				SetSelectionContext(SelectionContext::Type::INSTANCE, i);
		}

		MirroredVector<Light>& lights = m_Context->GetLights();
		for (uint32_t i = 0; i < lights.Size(); i++)
		{
			const Light& light = lights[i];
			if (light.type == Light::Type::MESH)
				continue;

			bool itemSelected = m_SelectionContext.type == SelectionContext::Type::LIGHT && m_SelectionContext.idx == i;
			ImGuiTreeNodeFlags flags = (itemSelected ? ImGuiTreeNodeFlags_Selected : 0) | LeafFlags;
			ImGui::TreeNodeEx(std::to_string(i).c_str(), flags, "%s %u", lightTypeNames[static_cast<int>(light.type)], i);

			if (ImGui::IsItemClicked())
				SetSelectionContext(SelectionContext::Type::LIGHT, i);
		}
		ImGui::PopStyleVar();

		// Clicking empty space clears the selection. IsItemClicked above fires on the same frame as
		// IsMouseDown, so without the IsAnyItemHovered guard this cleared the selection on the very
		// frame a row set it, and clicking a row did nothing.
		if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered())
			m_SelectionContext.idx = -1;

		ImGui::End();

		ImGui::Begin("Properties");
		if (m_SelectionContext.idx != -1)
			DrawProperties(m_SelectionContext);

		ImGui::End();
	}

	void SceneHierarchyPanel::DrawProperties(SelectionContext selectionContext)
	{
		if (selectionContext.type == SelectionContext::Type::INSTANCE)
		{
			auto meshInstance = m_Context->GetMeshInstances().Mutate(selectionContext.idx);

			if (UI::BeginSection("Transform"))
			{
				if (UI::BeginPropertyTable("transform"))
				{
					UI::DragFloat3Row("Location", meshInstance->position);
					ImGui::Spacing();
					UI::DragFloat3Row("Rotation", meshInstance->rotation);
					ImGui::Spacing();
					UI::DragFloat3Row("Scale", meshInstance->scale, 0.01f, "%.3f");

					UI::EndPropertyTable();
				}

				UI::EndSection();
			}

			AssetManager& assetManager = m_Context->GetAssetManager();

			MirroredVector<Material>& materials = assetManager.GetMaterials();
			std::string materialsString = assetManager.GetMaterialsString();

			if (UI::BeginSection("Material"))
			{
				if (meshInstance->materialIdx == -1)
				{
					if (ImGui::Button("Custom material"))
						meshInstance->materialIdx = 0;
				}
				else
				{
					if (UI::BeginPropertyTable("material"))
					{
						UI::PropertyLabel("Id");
						int materialIdx = meshInstance->materialIdx;
						if (ImGui::Combo("##id", &materialIdx, materialsString.c_str()))
							meshInstance->materialIdx = materialIdx;

						// Taken after the combo so it guards whichever material is now selected.
						// Specular and Emission take their own guards below: the sub-sections have
						// to sit outside this table, and a guard only records what it outlives.
						auto material = materials.Mutate(meshInstance->materialIdx);

						UI::PropertyLabel("Base color");
						ImGui::ColorEdit3("##basecolor", (float*)&material->baseColor, ImGuiColorEditFlags_NoInputs);
						UI::PropertyLabel("Metalness");
						ImGui::DragFloat("##metalness", &material->metalness, 0.01f, 0.0f, 1.0f);
						UI::PropertyLabel("Roughness");
						ImGui::DragFloat("##roughness", &material->roughness, 0.01f, 0.0f, 1.0f);
						UI::PropertyLabel("IOR", "Index of refraction");
						ImGui::DragFloat("##ior", &material->ior, 0.01f, 1.0f, 2.5f);
						UI::PropertyLabel("Transmission");
						ImGui::DragFloat("##transmission", &material->transmission, 0.01f, 0.0f, 1.0f);
						UI::PropertyLabel("Opacity");
						ImGui::DragFloat("##opacity", (float*)&material->opacity, 0.01f, 0.0f, 1.0f);

						UI::EndPropertyTable();
					}

					if (UI::BeginSection("Specular"))
					{
						auto material = materials.Mutate(meshInstance->materialIdx);
						if (UI::BeginPropertyTable("specular"))
						{
							UI::PropertyLabel("Weight");
							ImGui::DragFloat("##specularweight", &material->specularWeight, 0.01f, 0.0f, 1.0f);
							UI::PropertyLabel("Color");
							ImGui::ColorEdit3("##specularcolor", (float*)&material->specularColor, ImGuiColorEditFlags_NoInputs);
							UI::PropertyLabel("Anisotropy");
							ImGui::DragFloat("##anisotropy", &material->anisotropy, 0.01f, 0.0f, 1.0f);

							UI::EndPropertyTable();
						}
						UI::EndSection();
					}
					if (UI::BeginSection("Emission", false))
					{
						auto material = materials.Mutate(meshInstance->materialIdx);
						if (UI::BeginPropertyTable("emission"))
						{
							UI::PropertyLabel("Color");
							ImGui::ColorEdit3("##emissioncolor", (float*)&material->emissionColor, ImGuiColorEditFlags_NoInputs);
							UI::PropertyLabel("Intensity");
							ImGui::DragFloat("##emissionintensity", (float*)&material->intensity, 0.1f, 0.0f, 1000.0f);

							UI::EndPropertyTable();
						}
						UI::EndSection();
					}
				}
				UI::EndSection();
			}
		}
		else if (selectionContext.type == SelectionContext::Type::LIGHT)
		{
			auto light = m_Context->GetLights().Mutate(selectionContext.idx);
			if (UI::BeginSection("Light"))
			{
				if (UI::BeginPropertyTable("light"))
				{
					UI::PropertyLabel("Type");
					int currentIndex = static_cast<int>(light->type);
					if (ImGui::Combo("##type", &currentIndex, lightTypeNames, IM_ARRAYSIZE(lightTypeNames)))
						light->type = static_cast<Light::Type>(currentIndex);

					switch (light->type)
					{
					case Light::Type::POINT:
						UI::DragFloat3Row("Location", light->point.position);
						UI::PropertyLabel("Color");
						ImGui::ColorEdit3("##color", (float*)&light->point.color, ImGuiColorEditFlags_NoInputs);
						UI::PropertyLabel("Intensity");
						ImGui::DragFloat("##intensity", &light->point.intensity, 0.1f, 0.0f, 1000.0f);
						break;

					case Light::Type::SPOT:
						UI::DragFloat3Row("Location", light->spot.position);
						UI::PropertyLabel("Color");
						ImGui::ColorEdit3("##color", (float*)&light->spot.color, ImGuiColorEditFlags_NoInputs);
						UI::PropertyLabel("Intensity");
						ImGui::DragFloat("##intensity", &light->spot.intensity, 0.1f, 0.0f, 1000.0f);
						UI::PropertyLabel("Falloff start");
						ImGui::DragFloat("##falloffstart", &light->spot.falloffStart, 0.1f, 0.0f, 180.0f);
						UI::PropertyLabel("Falloff end");
						ImGui::DragFloat("##falloffend", &light->spot.falloffEnd, 0.1f, 0.0f, 180.0f);
						break;

					case Light::Type::DIRECTIONAL:
						UI::DragFloat3Row("Direction", light->directional.direction);
						UI::PropertyLabel("Color");
						ImGui::ColorEdit3("##color", (float*)&light->directional.color, ImGuiColorEditFlags_NoInputs);
						UI::PropertyLabel("Intensity");
						ImGui::DragFloat("##intensity", &light->directional.intensity, 0.1f, 0.0f, 1000.0f);
						break;
					default:
						break;
					}

					UI::EndPropertyTable();
				}
				UI::EndSection();
			}
		}
	}

}
