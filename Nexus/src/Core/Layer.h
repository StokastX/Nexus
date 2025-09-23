#pragma once

namespace Nexus {

	class Layer
	{
	public:
		virtual ~Layer() = default;

		virtual void OnAttach() {}
		virtual void OnDetach() {}

		virtual void OnEvent() {}

		virtual void OnUpdate(float deltaTime) {}
		virtual void OnRender() {}
	};

}