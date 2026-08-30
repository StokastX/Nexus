#include "Utils.h"

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

namespace Nexus::Utils {

	void GetPathAndFileName(const std::string fullPath, std::string& path, std::string& name)
	{
		int splitIndex = fullPath.find_last_of("/\\");
		name = fullPath.substr(splitIndex + 1);
		path = fullPath.substr(0, splitIndex + 1);
	}

}
