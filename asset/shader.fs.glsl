#version 420 core 

out vec4 color; 

uniform sampler2D tex;
uniform int ydim;

void main() 
{
	vec3 texel = texelFetch(tex, ivec2(gl_FragCoord.x, gl_FragCoord.y), 0).rgb;

	color = vec4(texel, 1.0f);
} 