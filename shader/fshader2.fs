#version 440 core
layout(location = 0) out vec4 lastframe;

in vec3 pos;

uniform sampler2D update;

void main() {
	lastframe = vec4(texture2D(update, 0.5 * pos.xy + 0.5).rgb, 1.0);
}