#include "Paths.h"

#include <cstdlib>

#ifndef NEXUS_ROOT_DIR
	// Defined by CMake. The fallback only matters for tooling that compiles this file outside the
	// build system, and keeps the old CWD-relative behaviour in that case.
	#define NEXUS_ROOT_DIR "."
#endif

namespace Nexus::Paths {

	static std::filesystem::path FindRoot()
	{
		// An explicit override wins, so the baked-in path is never a dead end.
#if defined(_MSC_VER)
		char* value = nullptr;
		size_t length = 0;
		if (_dupenv_s(&value, &length, "NEXUS_ROOT") == 0 && value)
		{
			std::filesystem::path root(value);
			free(value);
			return root;
		}
#else
		if (const char* value = std::getenv("NEXUS_ROOT"))
			return std::filesystem::path(value);
#endif
		return std::filesystem::path(NEXUS_ROOT_DIR);
	}

	const std::filesystem::path& Root()
	{
		static const std::filesystem::path root = FindRoot();
		return root;
	}

	std::string Resolve(const std::filesystem::path& relative)
	{
		if (relative.is_absolute())
			return relative.string();

		return (Root() / relative).string();
	}

}
