#version 140

in vec3 v_colour;
out vec4 color;

//uniform int u_colour_format;
uniform bool u_clipping;
uniform bool u_slice;
uniform float u_slice_width;

void main() {
    float z = gl_FragCoord.z;

    float clipping_dist = 0.5;

    // Cutaway
    if (u_clipping && (z <= clipping_dist || (u_slice && z >= clipping_dist + u_slice_width))) {
        discard;
    }
    vec2 pos = gl_PointCoord - vec2(0.5);
    // Shape of point
    if (dot(pos, pos) > 0.25) {
        discard;
    }

    // Normalise colours
    //float c = pow(2, u_colour_format * 8);

    // if (c == 0) {
    //     color = vec4(1.0);
    // } else {
    //     color = vec4(v_colour / c, 1.0);
    // }

    color = vec4(v_colour / 256.0, 1.0);
}
