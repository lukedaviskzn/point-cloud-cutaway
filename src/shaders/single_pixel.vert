#version 140

in vec3 position;
in vec3 colour;

out vec3 v_colour;

uniform mat4 u_modelview;
uniform mat4 u_projection;

void main() {
    v_colour = colour;

    vec4 pos = u_modelview * vec4(position, 1.0);
    
    gl_Position = u_projection * pos;
    gl_PointSize = 1;
}