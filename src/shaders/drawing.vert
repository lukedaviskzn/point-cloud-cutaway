#version 140

in vec3 position;

out vec3 v_position;

uniform mat4 u_mvp;

void main() {
    v_position = position;

    vec4 pos = u_mvp * vec4(position, 1.0);

    gl_Position = pos;
}