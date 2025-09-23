#version 460 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uModelInvTrans;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vPosWorld;

void main()
{
    vNormal = normalize(mat3(uModelInvTrans) * aNormal);

    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vPosWorld = worldPos.xyz;
    gl_Position = uProj * uView * worldPos;
}