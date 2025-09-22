#include "Shader.h"
#include <fstream>
#include <sstream>
#include "GL/glew.h"

namespace Nexus {

	Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath)
	{
		// 1. retrieve the vertex/fragment source code from filePath
		std::string vertexCode;
		std::string fragmentCode;
		std::ifstream vShaderFile;
		std::ifstream fShaderFile;
		// ensure ifstream objects can throw exceptions:
		vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		try
		{
			// open files
			vShaderFile.open(vertexPath);
			fShaderFile.open(fragmentPath);
			std::stringstream vShaderStream, fShaderStream;
			// read file's buffer contents into streams
			vShaderStream << vShaderFile.rdbuf();
			fShaderStream << fShaderFile.rdbuf();
			// close file handlers
			vShaderFile.close();
			fShaderFile.close();
			// convert stream into string
			vertexCode = vShaderStream.str();
			fragmentCode = fShaderStream.str();
		}
		catch (std::ifstream::failure& e)
		{
			std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
		}
		const char* vShaderCode = vertexCode.c_str();
		const char* fShaderCode = fragmentCode.c_str();
		// 2. compile shaders
		uint32_t vertex, fragment;
		// vertex shader
		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		CheckCompileErrors(vertex, "VERTEX");
		// fragment Shader
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		CheckCompileErrors(fragment, "FRAGMENT");
		// shader Program
		m_Id = glCreateProgram();
		glAttachShader(m_Id, vertex);
		glAttachShader(m_Id, fragment);
		glLinkProgram(m_Id);
		CheckCompileErrors(m_Id, "PROGRAM");
		// delete the shaders as they're linked into our program now and no longer necessary
		glDeleteShader(vertex);
		glDeleteShader(fragment);
	}

	void Shader::Use()
	{
		glUseProgram(m_Id);
	}

	void Shader::SetBool(const std::string& name, bool value)
	{
		glUniform1i(glGetUniformLocation(m_Id, name.c_str()), (int)value);
	}

	void Shader::SetInt(const std::string& name, int value)
	{
		glUniform1i(glGetUniformLocation(m_Id, name.c_str()), value);
	}

	void Shader::SetFloat(const std::string& name, float value)
	{
		glUniform1f(glGetUniformLocation(m_Id, name.c_str()), value);
	}

	void Shader::SetVec2(const std::string& name, const float2& value)
	{
		glUniform2fv(glGetUniformLocation(m_Id, name.c_str()), 1, &value.x);
	}

	void Shader::SetVec3(const std::string& name, const float3& value)
	{
		glUniform3fv(glGetUniformLocation(m_Id, name.c_str()), 1, &value.x);
	}

	void Shader::SetVec4(const std::string& name, const float4& value)
	{
		glUniform4fv(glGetUniformLocation(m_Id, name.c_str()), 1, &value.x);
	}

	void Shader::SetMat4(const std::string& name, const Mat4& mat)
	{
		// Transpose the matrix since Mat4 is row-major
		glUniformMatrix4fv(glGetUniformLocation(m_Id, name.c_str()), 1, GL_TRUE, &mat.cell[0]);
	}

	void Shader::CheckCompileErrors(uint32_t shader, const std::string& type)
	{
		int success;
		char infoLog[1024];
		if (type != "PROGRAM")
		{
			glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
			}
		}
		else
		{
			glGetProgramiv(shader, GL_LINK_STATUS, &success);
			if (!success)
			{
				glGetProgramInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
			}
		}
	}

}
