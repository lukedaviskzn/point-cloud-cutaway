#version 140

in vec3 v_colour;
out vec4 color;

uniform int u_colour_format;
uniform float u_clipping_dist;
uniform bool u_slice;

void main() {
    float z = gl_FragCoord.z / gl_FragCoord.w;
    // Cutaway
    if (z <= u_clipping_dist || (u_slice && z >= u_clipping_dist + 0.05)) {
        discard;
    }
    vec2 pos = gl_PointCoord - vec2(0.5);
    // Shape of point
    if (dot(pos, pos) > 0.25) {
        discard;
    }

    // Normalise colours
    float c = max(1, pow(2, u_colour_format * 8) - 1);

    color = vec4(v_colour / c, 1.0);
}
