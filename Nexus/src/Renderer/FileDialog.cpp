#include "FileDialog.h"

#include "tinyfiledialogs.h"

std::string FileDialog::OpenFile(const std::vector<const char*>& filters, const char* description)
{
    const char* file = tinyfd_openFileDialog(
        "Open File",
        "../",
        filters.size(),
        filters.data(),
        description,
        0
    );

    return file ? std::string(file) : std::string();
}

std::string FileDialog::SaveFile(const std::vector<const char*>& filters, const char* description)
{
    const char* file = tinyfd_saveFileDialog(
        "Save File",
        "../",
        filters.size(),
        filters.data(),
        description
    );

    return file ? std::string(file) : std::string();
}
