#version 460 core

layout(location = 0) in vec3 aPos;

uniform mat4 uView;
uniform mat4 uProj;

out vec2 TexCoord;
out vec3 vPos;

void main()
{
    gl_Position = uProj * uView * vec4(aPos, 1.0);
    vPos = aPos;
    TexCoord = aPos.xz;
}