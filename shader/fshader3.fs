#version 440 core

in vec3 pos;
out vec4 fragColor;

uniform sampler2D lastframe;

void main() {
	vec3 color = texture2D(lastframe, 0.5f * pos.xy + 0.5f).rgb;
	color = pow(color, vec3(1.0f / 2.2f));
	fragColor = vec4(color, 1.0f);
}