#version 140

out vec4 color;

uniform float u_slice_width;

void main() {
    float z = gl_FragCoord.z / gl_FragCoord.w;

    float clipping_dist = 0.5;
    
    // Cutaway
    if (z <= clipping_dist || z >= clipping_dist + u_slice_width) {
        discard;
    }
    vec2 pos = gl_PointCoord - vec2(0.5);

    color = vec4(1.0, 1.0, 1.0, 0.0);
}
