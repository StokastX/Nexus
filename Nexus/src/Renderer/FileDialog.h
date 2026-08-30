#pragma once

#include <iostream>
#include <vector>

namespace Nexus {

	class FileDialog
	{
	public:
		static std::string OpenFile(const std::vector<const char*>& filters, const char* description);
		static std::string SaveFile(const std::vector<const char*>& filters, const char* description);
	};

}
