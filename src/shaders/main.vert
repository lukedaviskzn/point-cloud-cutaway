#version 140

in vec3 position;
in vec3 colour;
in float size;

out vec3 v_colour;

// uniform mat4 mvp;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_perspective;

void main() {
    v_colour = colour;
    vec4 pos = u_view * u_model * vec4(position, 1.0);
    
    // gl_Position = mvp * vec4(position, 1.0);
    gl_Position = u_perspective * pos;
    gl_PointSize = size / length(vec3(pos)) * 200;
}