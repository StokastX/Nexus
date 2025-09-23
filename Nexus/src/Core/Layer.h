#pragma once
#include "Events/Event.h"

namespace Nexus {

	class Layer
	{
	public:
		virtual ~Layer() = default;

		virtual void OnAttach() {}
		virtual void OnDetach() {}

		virtual void OnEvent(Event& e) {}

		virtual void OnUpdate(float deltaTime) {}
		virtual void OnRender() {}
	};

}