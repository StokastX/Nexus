#pragma once
#include <cstdint>

/*
 * Move-only owner of a raw OpenGL object name.
 *
 * OpenGL objects are bare uint32_t names, so a class holding one has to hand-write a destructor
 * and a move constructor, and remember to delete the copy constructor. Forgetting the last part
 * is silent: the copy duplicates the name and the first destructor to run deletes an object the
 * other copy still points at. Wrapping the name in a GLHandle makes that a compile error instead.
 *
 * The deleter is a template parameter so the handle type itself carries what kind of object it
 * owns, and no vtable or std::function is involved.
 */
namespace Nexus {

	// Adapters over the glDelete* entry points, which take a count and an array.
	// Defined in the .cpp so that including this header does not pull in glew.
	namespace GLDelete {

		void Buffer(uint32_t handle);
		void VertexArray(uint32_t handle);
		void Texture(uint32_t handle);
		void Framebuffer(uint32_t handle);
		void Renderbuffer(uint32_t handle);
		void Program(uint32_t handle);

	}

	template<void (*Deleter)(uint32_t)>
	class GLHandle
	{
	public:
		GLHandle() = default;
		explicit GLHandle(uint32_t handle) : m_Handle(handle) {}

		~GLHandle() { Reset(); }

		GLHandle(const GLHandle&) = delete;
		GLHandle& operator=(const GLHandle&) = delete;

		GLHandle(GLHandle&& other) noexcept
			: m_Handle(other.m_Handle)
		{
			other.m_Handle = 0;
		}

		GLHandle& operator=(GLHandle&& other) noexcept
		{
			if (this != &other)
			{
				Reset();
				m_Handle = other.m_Handle;
				other.m_Handle = 0;
			}
			return *this;
		}

		// 0 is OpenGL's "no object" name, so it doubles as the empty state.
		// Deliberately no implicit conversion to uint32_t: call sites spell out Get(), so a
		// handle can never silently decay into an int and get stored past its owner's lifetime.
		uint32_t Get() const { return m_Handle; }
		explicit operator bool() const { return m_Handle != 0; }

		// Address of the name, for the glGen*/glCreate* entry points that write through a pointer.
		uint32_t* AddressOf() { return &m_Handle; }

		void Reset(uint32_t handle = 0)
		{
			if (m_Handle)
				Deleter(m_Handle);
			m_Handle = handle;
		}

		// Relinquishes ownership without deleting.
		uint32_t Release()
		{
			uint32_t handle = m_Handle;
			m_Handle = 0;
			return handle;
		}

	private:
		uint32_t m_Handle = 0;
	};

	using GLBufferHandle = GLHandle<GLDelete::Buffer>;
	using GLVertexArrayHandle = GLHandle<GLDelete::VertexArray>;
	using GLTextureHandle = GLHandle<GLDelete::Texture>;
	using GLFramebufferHandle = GLHandle<GLDelete::Framebuffer>;
	using GLRenderbufferHandle = GLHandle<GLDelete::Renderbuffer>;
	using GLProgramHandle = GLHandle<GLDelete::Program>;

}
