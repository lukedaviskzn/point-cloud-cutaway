#version 140

in vec3 position;

out vec3 v_position;

void main() {
    v_position = position;

    gl_Position = vec4(position, 1.0);
}