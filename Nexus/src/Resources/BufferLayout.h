#pragma once
#include <cstdint>
#include <vector>

/*
 * Backend-neutral description of the contents of a vertex buffer: which attributes it holds, in
 * what order, and how they are packed. A graphics backend turns this into its own attribute setup
 * calls; nothing here is OpenGL-specific.
 */
namespace Nexus {

	enum class ShaderDataType
	{
		None = 0,
		Float, Float2, Float3, Float4,
		Int, Int2, Int3, Int4,
		UInt, UInt2, UInt3, UInt4,
		UByte4,
		Mat3, Mat4
	};

	/*
	 * The three properties below are the primitive ones, and total size is derived from them.
	 * Writing size as a switch of its own is how the component count and the byte size drift
	 * apart when a type is added to one and forgotten in the other.
	 *
	 * None of them have a default case, so adding an enumerator makes the compiler point at
	 * every switch that has to handle it.
	 */

	// Size in bytes of one component (a Float3 is three 4-byte components).
	constexpr uint32_t ShaderDataTypeComponentSize(ShaderDataType type)
	{
		switch (type)
		{
			case ShaderDataType::Float:  case ShaderDataType::Float2:
			case ShaderDataType::Float3: case ShaderDataType::Float4:
			case ShaderDataType::Mat3:   case ShaderDataType::Mat4:
			case ShaderDataType::Int:    case ShaderDataType::Int2:
			case ShaderDataType::Int3:   case ShaderDataType::Int4:
			case ShaderDataType::UInt:   case ShaderDataType::UInt2:
			case ShaderDataType::UInt3:  case ShaderDataType::UInt4:
				return 4;
			case ShaderDataType::UByte4:
				return 1;
			case ShaderDataType::None:
				return 0;
		}
		return 0;
	}

	// Components in a single attribute slot. A Mat4 is four slots of four components, not sixteen.
	constexpr uint32_t ShaderDataTypeComponentCount(ShaderDataType type)
	{
		switch (type)
		{
			case ShaderDataType::Float:  case ShaderDataType::Int:  case ShaderDataType::UInt:
				return 1;
			case ShaderDataType::Float2: case ShaderDataType::Int2: case ShaderDataType::UInt2:
				return 2;
			case ShaderDataType::Float3: case ShaderDataType::Int3: case ShaderDataType::UInt3:
			case ShaderDataType::Mat3:
				return 3;
			case ShaderDataType::Float4: case ShaderDataType::Int4: case ShaderDataType::UInt4:
			case ShaderDataType::UByte4: case ShaderDataType::Mat4:
				return 4;
			case ShaderDataType::None:
				return 0;
		}
		return 0;
	}

	// Consecutive attribute locations the type consumes. Only matrices need more than one:
	// a mat4 vertex attribute is passed as four separate vec4 locations.
	constexpr uint32_t ShaderDataTypeSlotCount(ShaderDataType type)
	{
		switch (type)
		{
			case ShaderDataType::Mat3: return 3;
			case ShaderDataType::Mat4: return 4;
			default: return 1;
		}
	}

	// Total size in bytes, derived rather than duplicated.
	constexpr uint32_t ShaderDataTypeSize(ShaderDataType type)
	{
		return ShaderDataTypeComponentSize(type)
			* ShaderDataTypeComponentCount(type)
			* ShaderDataTypeSlotCount(type);
	}

	// True for types fed to the shader as genuine integers rather than converted to float.
	// UByte4 is deliberately excluded: it exists for normalised data such as packed colours.
	constexpr bool ShaderDataTypeIsInteger(ShaderDataType type)
	{
		switch (type)
		{
			case ShaderDataType::Int:  case ShaderDataType::Int2:
			case ShaderDataType::Int3: case ShaderDataType::Int4:
			case ShaderDataType::UInt: case ShaderDataType::UInt2:
			case ShaderDataType::UInt3: case ShaderDataType::UInt4:
				return true;
			default:
				return false;
		}
	}

	struct BufferElement
	{
		// Documents the attribute at the call site. The backends do not read it: attributes are
		// matched to the shader by location, not by name.
		const char* Name = nullptr;
		ShaderDataType Type = ShaderDataType::None;
		uint32_t Size = 0;
		uint32_t Offset = 0;
		bool Normalized = false;

		constexpr BufferElement() = default;

		constexpr BufferElement(ShaderDataType type, const char* name, bool normalized = false)
			: Name(name), Type(type), Size(ShaderDataTypeSize(type)), Offset(0), Normalized(normalized)
		{
		}

		constexpr uint32_t GetComponentCount() const { return ShaderDataTypeComponentCount(Type); }
		constexpr uint32_t GetSlotCount() const { return ShaderDataTypeSlotCount(Type); }

		// Byte distance between the consecutive slots of a matrix attribute.
		constexpr uint32_t GetSlotSize() const
		{
			return ShaderDataTypeComponentSize(Type) * ShaderDataTypeComponentCount(Type);
		}
	};

	class BufferLayout
	{
	public:
		BufferLayout() = default;

		BufferLayout(std::initializer_list<BufferElement> elements)
			: m_Elements(elements)
		{
			CalculateOffsetsAndStride();
		}

		uint32_t GetStride() const { return m_Stride; }
		const std::vector<BufferElement>& GetElements() const { return m_Elements; }
		bool IsEmpty() const { return m_Elements.empty(); }

		// Const iteration only. Handing out mutable elements would let a caller change a Type
		// after construction, leaving the cached offsets and stride describing the old layout.
		std::vector<BufferElement>::const_iterator begin() const { return m_Elements.begin(); }
		std::vector<BufferElement>::const_iterator end() const { return m_Elements.end(); }

	private:
		void CalculateOffsetsAndStride()
		{
			uint32_t offset = 0;
			for (BufferElement& element : m_Elements)
			{
				element.Offset = offset;
				offset += element.Size;
			}
			m_Stride = offset;
		}

		std::vector<BufferElement> m_Elements;
		uint32_t m_Stride = 0;
	};

}
