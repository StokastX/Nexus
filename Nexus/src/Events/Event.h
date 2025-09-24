#pragma once
#include <iostream>

/*
 * This event system is adapted from the Cherno's game engine series.
**/
namespace Nexus {

	enum class EventType
	{
		None = 0,
		KeyPressed, KeyReleased, KeyTyped,
		MouseButtonPressed, MouseButtonReleased, MouseMoved, MouseScrolled,
		WindowResized, WindowClosed
	};

	enum EventCategory
	{
		None = 0,
		EventCategoryInput		 = (1 << 0),
		EventCategoryKeyboard	 = (1 << 1),
		EventCategoryMouse		 = (1 << 2),
		EventCategoryMouseButton = (1 << 3),
		EventCategoryApplication = (1 << 4),
	};

#define EVENT_CLASS_TYPE(type) static EventType GetStaticType() { return EventType::type; }\
								virtual EventType GetEventType() const override { return GetStaticType(); }\
								virtual const char* GetName() const override { return #type; }

#define EVENT_CLASS_CATEGORY(category) virtual int GetCategoryFlags() const override { return category; }

	class Event
	{
	public:
		virtual ~Event() = default;

		virtual EventType GetEventType() const = 0;
		virtual const char* GetName() const = 0;
		virtual int GetCategoryFlags() const = 0;
		virtual std::string ToString() const { return GetName(); }

		bool IsInCategory(EventCategory category)
		{
			return GetCategoryFlags() & category;
		}

	public:
		bool handled = false;
	};

	class EventDispatcher
	{
	public:
		EventDispatcher(Event& event)
			: m_Event(event)
		{
		}

		template <typename T, typename F>
		bool Dispatch(const F& func)
		{
			if (m_Event.GetEventType() == T::GetStaticType())
			{
				m_Event.handled |= func(static_cast<T&>(m_Event));
				return true;
			}
			return false;
		}

	private:
		Event& m_Event;
	};

}