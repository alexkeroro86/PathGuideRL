#pragma once

#include "cuda_helper.cuh"

#ifndef GLUT_KEY_ESC
#define GLUT_KEY_ESC 0x001B
#endif

class glshader
{
public:
	GLuint mID;

	glshader() {}
	glshader(const char *vert_path, const char *frag_path, const char *geom_path = nullptr)
	{
		GLuint vert_shader, frag_shader, geom_shader;

		mID = glCreateProgram();

		// vertex shader
		vert_shader = glCreateShader(GL_VERTEX_SHADER);
		const GLchar* vert_source = ReadTextFile(vert_path);

		glShaderSource(vert_shader, 1, &vert_source, NULL);
		delete[] vert_source;
		glCompileShader(vert_shader);
		CheckStatusError("SHADER", vert_shader, "VERTEX");
		glAttachShader(mID, vert_shader);

		// fragment shader
		frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
		const GLchar* frag_source = ReadTextFile(frag_path);

		glShaderSource(frag_shader, 1, &frag_source, NULL);
		delete[] frag_source;
		glCompileShader(frag_shader);
		CheckStatusError("SHADER", frag_shader, "FRAGMENT");
		glAttachShader(mID, frag_shader);

		// geometry shader (optional)
		if (geom_path != nullptr) {
			geom_shader = glCreateShader(GL_GEOMETRY_SHADER);
			const GLchar* geom_source = ReadTextFile(geom_path);

			glShaderSource(geom_shader, 1, &geom_source, NULL);
			delete[] geom_source;
			glCompileShader(geom_shader);
			CheckStatusError("SHADER", geom_shader, "GEOMETRY");
			glAttachShader(mID, geom_shader);
		}

		// program
		glLinkProgram(mID);
		CheckStatusError("PROGRAM", mID);

		glDeleteShader(vert_shader);
		glDeleteShader(frag_shader);
		if (geom_path != nullptr) {
			glDeleteShader(geom_shader);
		}
	}
	void use()
	{
		glUseProgram(mID);
	}
	// default uniform
	void setBool(const std::string &name, bool value) const
	{
		glUniform1i(glGetUniformLocation(mID, name.c_str()), (int)value);
	}
	void setInt(const std::string &name, int value) const
	{
		glUniform1i(glGetUniformLocation(mID, name.c_str()), value);
	}
	void setFloat(const std::string &name, float value) const
	{
		glUniform1f(glGetUniformLocation(mID, name.c_str()), value);
	}
	void setVec2(const std::string &name, float x, float y) const
	{
		glUniform2f(glGetUniformLocation(mID, name.c_str()), x, y);
	}
	void setVec3(const std::string &name, float x, float y, float z) const
	{
		glUniform3f(glGetUniformLocation(mID, name.c_str()), x, y, z);
	}
	void setVec4(const std::string &name, float x, float y, float z, float w)
	{
		glUniform4f(glGetUniformLocation(mID, name.c_str()), x, y, z, w);
	}

private:
	// helper function
	const GLchar* ReadTextFile(const char* path)
	{
		std::string code;
		std::ifstream file;

		FILE* fp;
		errno_t err = fopen_s(&fp, path, "rb");

		if (err != 0) {
			std::cerr << "UNABLE TO OPEN FILE: " << path << std::endl;
			exit(EXIT_FAILURE);
		}

		fseek(fp, 0, SEEK_END);
		int len = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		GLchar* source = new GLchar[len + 1];

		fread(source, 1, len, fp);
		fclose(fp);

		source[len] = 0;
		return const_cast<const GLchar*>(source);
	}

	void CheckStatusError(const char* type, GLuint target, const char *sub_type = "")
	{
		GLint success;
		if (strcmp(type, "SHADER") == 0) {
			glGetShaderiv(target, GL_COMPILE_STATUS, &success);
			if (!success) {
				GLsizei len;
				glGetShaderiv(target, GL_INFO_LOG_LENGTH, &len);
				GLchar* log = new GLchar[len + 1];
				glGetShaderInfoLog(target, len, NULL, log);
				std::cerr << "ERROR::" << sub_type << "_SHADER_COMPILE_ERROR\n" << log << std::endl;
			}
		}
		else if (strcmp(type, "PROGRAM") == 0) {
			glGetProgramiv(target, GL_LINK_STATUS, &success);
			if (!success) {
				GLsizei len;
				glGetProgramiv(target, GL_INFO_LOG_LENGTH, &len);
				GLchar* log = new GLchar[len + 1];
				glGetProgramInfoLog(target, len, NULL, log);
				std::cerr << "ERROR::PROGRAM_LINK_ERROR\n" << log << std::endl;
			}
		}
		else {
			std::cerr << "ERROR::INVALID_TYPE\n" << std::endl;
		}
	}
};

