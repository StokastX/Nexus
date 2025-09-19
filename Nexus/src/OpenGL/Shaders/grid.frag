#version 330 core

in vec2 TexCoord;
in vec3 vPos;

uniform vec3 uCamPosWorld;

out vec4 FragColor;

const float MinorLineWidth = 4;
const float MajorLineWidth = 4;
const float AxisLineWidth = 4;
const float MajorGridDiv = 10;
const float GridHorizonDist = 100.0;
const vec4 yAxisColor = vec4(0.396, 0.549, 0.145, 1.0);
const vec4 xAxisColor = vec4(0.612, 0.235, 0.29, 1.0);
const vec4 minorLineColor = vec4(0.34, 0.34, 0.34, 1.0);
const vec4 majorLineColor = vec4(0.38, 0.38, 0.38, 1.0);
const vec4 baseColor = vec4(0.0, 0.0, 0.0, 0.0);


// Antialiased grid shader adapted from https://gist.github.com/bgolus/3a561077c86b5bfead0d6cc521097bae
void main()
{
    vec2 ddx = dFdx(TexCoord);
    vec2 ddy = dFdy(TexCoord);
    vec2 uvDeriv = vec2(length(vec2(ddx.x, ddy.x)), length(vec2(ddx.y, ddy.y)));

    vec2 axisDrawWidth = uvDeriv * AxisLineWidth;
    vec2 axisLineAA = uvDeriv * 1.5;
    vec2 axisLines2 = smoothstep(axisDrawWidth + axisLineAA, axisDrawWidth - axisLineAA, abs(TexCoord * 2.0));

    vec2 majorUVDeriv = uvDeriv / MajorGridDiv;
	float majorLineWidth = MajorLineWidth / MajorGridDiv;
	vec2 majorDrawWidth = uvDeriv * majorLineWidth;
	vec2 majorLineAA = majorUVDeriv * 1.5;
	vec2 majorGridUV = 1.0 - abs(fract(TexCoord / MajorGridDiv) * 2.0 - 1.0);
	vec2 majorAxisOffset = (1.0 - clamp(abs(TexCoord / MajorGridDiv * 2.0), 0.0, 1.0)) * 2.0;
	majorGridUV += majorAxisOffset; // adjust UVs so center axis line is skipped
	vec2 majorGrid2 = smoothstep(majorDrawWidth + majorLineAA, majorDrawWidth - majorLineAA, majorGridUV);
	majorGrid2 = clamp(majorGrid2 - axisLines2, 0.0, 1.0); // hack

	float minorTargetWidth = MinorLineWidth;
	vec2 minorDrawWidth = uvDeriv * MinorLineWidth;
	vec2 minorLineAA = uvDeriv * 1.5;
	vec2 minorGridUV = 1.0 - abs(fract(TexCoord) * 2.0 - 1.0);
	vec2 minorMajorOffset = (1.0 - clamp((1.0 - abs(fract(TexCoord / MajorGridDiv) * 2.0 - 1.0)) * MajorGridDiv, 0.0, 1.0)) * 2.0;
	minorGridUV += minorMajorOffset; // adjust UVs so major division lines are skipped
	vec2 minorGrid2 = smoothstep(minorDrawWidth + minorLineAA, minorDrawWidth - minorLineAA, minorGridUV);
	minorGrid2 = clamp(minorGrid2 - axisLines2, 0.0, 1.0); // hack

	float minorGrid = mix(minorGrid2.x, 1.0, minorGrid2.y);
	float majorGrid = mix(majorGrid2.x, 1.0, majorGrid2.y);

	vec4 axisLines = mix(xAxisColor * axisLines2.y, yAxisColor, axisLines2.x);

	vec4 col = mix(baseColor, minorLineColor, minorGrid *  minorLineColor.a);
	col = mix(col, majorLineColor, majorGrid * majorLineColor.a);
	col = col * (1.0 - axisLines.a) + axisLines;

	// Blender fading from https://github.com/blender/blender/blob/main/source/blender/draw/engines/overlay/shaders/overlay_grid_frag.glsl
	vec3 V = uCamPosWorld - vPos;
    float dist = length(V);
	V /= dist;
	float angle = 1.0 - abs(V.y);
	angle *= angle;
	float fade = 1.0 - angle * angle;
	fade *= 1.0 - smoothstep(0.0, GridHorizonDist, dist - GridHorizonDist);
	col.a *= fade;

	FragColor = vec4(col);
}
