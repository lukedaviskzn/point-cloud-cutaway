#version 140

in vec3 v_position;

out vec4 color;

uniform sampler2D u_cutaway;
uniform sampler2D u_cutaway_slice;
uniform sampler2D u_cutaway_slice_processed;

void main() {
    vec2 tex_coords = (v_position.xy + vec2(1.0, 1.0)) / 2.0;

    // color = vec4(tex_coords, 1.0, 1.0);

    vec4 cutaway_colour = texture(u_cutaway, tex_coords);
    vec4 slice_colour = texture(u_cutaway_slice, tex_coords);

    float slice_factor = slice_colour.a / 2 + 0.5;
    float cutaway_factor = 1.0 - slice_factor;

    color = vec4(cutaway_colour.rgb * cutaway_factor + slice_colour.rgb * slice_factor, 1.0);
}
