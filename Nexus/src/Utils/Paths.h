#pragma once

#include <filesystem>
#include <string>

namespace Nexus::Paths {

	/*
	 * Nexus loads data files at runtime (GLSL shaders, fonts, demo scenes) and those files live in
	 * the source tree, not next to the executable. Resolving them against the current working
	 * directory makes the build depend on how it was launched -- Visual Studio, VS Code and a
	 * double-click all use a different CWD -- so every such path goes through Resolve() instead.
	 */

	// Directory the runtime data is rooted at, i.e. the Nexus/ library directory. Baked in at
	// configure time by CMake; override it with the NEXUS_ROOT environment variable to point a
	// relocated build at its own copy of the data.
	const std::filesystem::path& Root();

	// Resolves a Root()-relative path. Absolute paths are passed through untouched.
	std::string Resolve(const std::filesystem::path& relative);

}
