#version 460 core

in vec3 vNormal;
in vec3 vPosWorld;

uniform vec3 uLightDirWorld;
uniform vec3 uCamPosWorld;

uniform bool uOutline;

uniform vec4 uPickingColor;
uniform bool uWritePicking;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 InstanceColor;

void main()
{
    vec3 N = vNormal;
	if (!gl_FrontFacing)
		N = -N;

    vec3 L = uLightDirWorld;
    vec3 V = normalize(uCamPosWorld - vPosWorld);

    // Diffuse
    float diffuse = max(dot(N, L), 0.0);

    // Specular (Blinn-Phong)
    vec3 H = normalize(L + V);
    float specular = pow(max(dot(N, H), 0.0), 32.0);

    // Weights
    float ambient = 0.2;
    float kd = 0.4; // diffuse factor
    float ks = 0.1; // specular factor

    float intensity = ambient + kd * diffuse + ks * specular;

    if (!uOutline)
		FragColor = vec4(vec3(intensity), 1.0);
    else
        FragColor = vec4(1.0, 0.62, 0.17, 1.0);

    if (uWritePicking)
    {
		InstanceColor = uPickingColor;
	}

}