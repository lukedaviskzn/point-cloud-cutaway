#version 140

in vec3 position;
in vec3 colour;
// in float size;

out vec3 v_colour;

uniform mat4 u_modelview;
uniform mat4 u_projection;
uniform float u_zoom;
uniform float u_size;

void main() {
    v_colour = colour;

    vec4 pos = u_modelview * vec4(position, 1.0);
    
    gl_Position = u_projection * pos;
    gl_PointSize = 1;
    // h = window height, d = size, z = dist to camera
    // s = 2*h*arctan(d/2z) / fovy ~= h*d/(z*fovy)
    //gl_PointSize = u_window_height*size/(pos.z*u_fovy);
    gl_PointSize = u_size * u_zoom;
}