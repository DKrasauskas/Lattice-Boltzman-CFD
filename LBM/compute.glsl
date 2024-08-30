#version 430 core
precision highp float;

layout(local_size_x = 10, local_size_y = 10) in;
layout(rgba32f, binding = 0) uniform image2D img_output;

layout(location = 2) uniform float max_p = 1;
layout(location = 3) uniform int min_p = 10;

//define x^3 - 1
//define x = vec2

layout(std430, binding = 1) buffer rx
{
	float datax[];
};
layout(std430, binding = 2) buffer ry
{
	float datay[];
};

vec4 color_map(float data) {
	//data -= 0.01f;
	data = abs(data);
	if (data < 0.166f) {
		return vec4(0.0, 0.0, data * 6, 1.0f);
	}
	if (data < 0.333f) {
		return vec4(0.0, (data - 0.1666f) * 6, 1.0f,  1.0);
	}
	if (data < 0.5f) {
		return vec4(0.0,  1.0, 1.0 - (data - 0.33f) * 6, 1.0);
	}
	if (data < 0.666f) {
		return vec4((data - 0.5f) * 6, 1.0, 0.0f, 1.0);
	}
	if (data < 0.8333f) {
		return vec4(1.0, 1.0 - (data - 0.66f) * 6, 0.0, 1.0);
	}
	if (data < 1.0f) {
		return vec4(1.0 - (data - 0.833f) * 6, 0.0f, 0.0, 1.0);
	}
	return vec4(1, 1, 1, 1);
}
vec4 color_map2(float data) {
	data = abs(data);
	if (data < 0.166f) {
		return vec4(0.0, data * 6, 0.0, 1.0f);
	}
	if (data < 0.333f) {
		return vec4(0.0, 1.0f, (data - 0.1666f) * 6, 1.0);
	}
	if (data < 0.5f) {
		return vec4(0.0, 1.0 - (data - 0.33f) * 6, 1.0, 1.0);
	}
	if (data < 0.666f) {
		return vec4((data - 0.5f) * 6, 0.0, 1.0f, 1.0);
	}
	if (data < 0.8333f) {
		return vec4(1.0, 0.0f, 1.0 - (data - 0.66f) * 6, 1.0);
	}
	if (data < 1.0f) {
		return vec4(1.0 - (data - 0.833f) * 6, 0.0f, 0.0, 1.0);
	}
	return vec4(0, 0, 1, 1);
}
uint n = 600;
uniform float w = 2;
uniform float ws = 2;
void main() {
	int rx = gl_GlobalInvocationID.x % 2 == 0 ? 1 : 0;
	int ry = (gl_GlobalInvocationID.y + 1) % 2 == 0 ? 1 : 0;
	float c = (rx == 1 && ry == 1) ? 1.0f : 0.1f;
	float theta = 0.1f;
	float cx = (gl_GlobalInvocationID.x * 1.0f - 512.0f * 0.25f) * cos(theta) - (gl_GlobalInvocationID.y * 1.0f - 256.0f * 0.5f) * sin(theta);
	float cy = (gl_GlobalInvocationID.x - 512.0f * 0.25f) * sin(theta) + (gl_GlobalInvocationID.y - 256.0f*0.5f) * cos(theta);
	float dx = (cx) * 0.006f;
	float data = datax[gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * n];
	vec4 color =  w >= 1.0f ? color_map2(datax[gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * n]) : color_map(datax[gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * n]);
	
	//vec4(datax[gl_GlobalInvocationID.x + gl_GlobalInvocationID.y *n], datax[gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * n], datax[gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * n], 1.0);
	/*if (dx >= 0 && dx < 1) {
		float dy = 0.2969 * sqrt(dx) - 0.126 * dx - 0.3516 * dx * dx + 0.2843 * dx * dx * dx - 0.1015 * dx * dx * dx * dx;
		if (abs(dy) > 0.01f * abs(cy))color = vec4(.1, .2, 0.2, 0);
	}*/
	if ((gl_GlobalInvocationID.x * 1.0f - 300/4) * (gl_GlobalInvocationID.x * 1.0f - 300/4) + (gl_GlobalInvocationID.y * 1.0f - 100) * (gl_GlobalInvocationID.y * 1.0f - 100) < 100) {
		color = vec4(0.0, .5, 1.0, 1);
	}
	imageStore(img_output, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y), color);
}