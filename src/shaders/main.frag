#version 140

in vec3 v_colour;
out vec4 color;

void main() {
    // Shape of point
    if (length(gl_PointCoord - vec2(0.5)) > 0.5) {
        discard;
    }

    // color = vec4(colour, 1.0);
    color = vec4(v_colour, 1.0);
}
