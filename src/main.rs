#[macro_use] extern crate glium;
#[macro_use] extern crate maplit;

use std::{sync::mpsc, thread};

use glium::{glutin::{self, event::{VirtualKeyCode, MouseButton, ElementState}, dpi::PhysicalPosition}, Surface, program::ProgramCreationInput};
use las::{Reader, Read};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use clap::Parser;

use crate::input::KeyboardManager;

mod input;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    colour: [f32; 3],
    size: f32,
}

#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long, value_parser)]
    file: String,
    #[clap(short, long, value_parser, default_value_t = 10.0)]
    point_size: f32
}

const FPS: f32 = 60.0;
const FRAME_LENGTH: f32 = 1.0/FPS;

fn main() {
    let args = Args::parse();
    let filename = args.file;
    let default_point_size = args.point_size;

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new()
        .with_gl_profile(glutin::GlProfile::Core)
        .with_multisampling(4);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    implement_vertex!(Vertex, position, colour, size);

    let mut camera_position: glam::Vec3 = glam::Vec3::ZERO;
    let mut camera_rotation: glam::Vec2 = glam::Vec2::ZERO;

    let mut mouse_position = glam::Vec2::NAN;

    let mut mouse_locked = false;

    // Flip y and z
    let coordinate_system_matrix = glam::mat4(
        glam::vec4(1.0, 0.0, 0.0, 0.0),
        glam::vec4(0.0, 0.0, 1.0, 0.0),
        glam::vec4(0.0, 1.0, 0.0, 0.0),
        glam::vec4(0.0, 0.0, 0.0, 1.0),
    );

    let mut keyboard = KeyboardManager::new();

    // let filename = "data/autzen.laz";
        
    let mut shape = vec![];

    let (tx, rx) = mpsc::channel();

    let mut reader = Reader::from_path(filename).unwrap();

    let n = reader.header().number_of_points();
    
    thread::spawn(move || {
        let mut i = 0;

        println!("Loading {} points", n);

        let mut last_progress = 0;

        while let Some(Ok(point)) = reader.read() {
            let colour = if let Some(colour) = point.color {
                [colour.red as f32 / 255.0, colour.green as f32 / 255.0, colour.blue as f32 / 255.0]
            } else {
                [1.0, 1.0, 1.0]
            };

            let v = Vertex {
                position: [point.x as f32, point.y as f32, point.z as f32],
                colour: colour,
                size: default_point_size,
            };

            tx.send(v).unwrap();

            i += 1;

            if i > n {
                break;
            }

            let progress = (100 * i) / n; // percentage
            if progress != last_progress {
                last_progress = progress;
                println!("Loading... {}%", progress);
            }
        }

        println!("Points Loaded");
    });

    let num_first_points = n * 5 / 100;

    // Wait for first 5% of points
    for _ in 0..num_first_points {
        shape.push(rx.recv().unwrap());
    }

    // Calculating totals
    let centre = shape.par_iter()
        .map(|x| x.position)
        .reduce(
            || [0.0, 0.0, 0.0],
            |x, y| [x[0]+y[0], x[1]+y[1], x[2]+y[2]],
        );
    
    // Calculate centre
    let centre = glam::vec3(centre[0], centre[1], centre[2]) / num_first_points as f32;

    let mut vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::Points);

    let vertex_shader_src = include_str!("shaders/main.vert");
    let fragment_shader_src = include_str!("shaders/main.frag");

    let program = glium::Program::new(&display, ProgramCreationInput::SourceCode {
        vertex_shader: vertex_shader_src,
        fragment_shader: fragment_shader_src,
        uses_point_size: true,
        tessellation_control_shader: None,
        tessellation_evaluation_shader: None,
        geometry_shader: None,
        transform_feedback_varyings: None,
        outputs_srgb: true,
    }).unwrap();
    
    event_loop.run(move |event, _, control_flow| {
        let next_frame_time = std::time::Instant::now() +
            std::time::Duration::from_nanos((FRAME_LENGTH * 1.0e9) as u64);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                glutin::event::WindowEvent::KeyboardInput { input, .. } => {
                    keyboard.update(input);

                    if let Some(key) = input.virtual_keycode {
                        if key == VirtualKeyCode::Escape && input.state == ElementState::Pressed {
                            let gl_window = display.gl_window();
                            let window = gl_window.window();
                            
                            let _ = window.set_cursor_grab(glutin::window::CursorGrabMode::None);
                            let _ = window.set_cursor_visible(true);

                            mouse_locked = false;
                        }
                    }

                    return;
                },
                glutin::event::WindowEvent::MouseInput { button, state, .. } => {
                    if button == MouseButton::Left && state == ElementState::Pressed {
                        let gl_window = display.gl_window();
                        let window = gl_window.window();
                        
                        let _ = window.set_cursor_grab(glutin::window::CursorGrabMode::Confined);
                        let _ = window.set_cursor_visible(false);
                        let size = display.gl_window().window().inner_size();
                        let _ = window.set_cursor_position(PhysicalPosition::new(size.width / 2, size.height / 2));
                        
                        mouse_position = glam::Vec2::NAN;

                        mouse_locked = true;
                    }
                    return;
                }
                glutin::event::WindowEvent::CursorMoved { position, .. } => {
                    mouse_position = glam::vec2(position.x as f32, position.y as f32);
                }
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        let mut target = display.draw();

        // Handle Update
        {
            let mut changed = false;
            // Get any newly available points
            while let Ok(point) = rx.try_recv() {
                shape.push(point);
                changed = true;
            }
            if changed {
                vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
            }

            let (width, height) = target.get_dimensions();
            let window_centre = glam::vec2(width as f32 / 2.0, height as f32 / 2.0);
            
            let mouse_delta = if !mouse_position.is_nan() && mouse_locked {
                let _ = display.gl_window().window().set_cursor_position(PhysicalPosition::new(width / 2, height / 2));
                mouse_position - window_centre
            } else {
                glam::Vec2::ZERO
            };

            let speed = 100.0; // units per second
            let angular_speed = 0.1; // radians per second
            let forward = glam::Quat::from_euler(glam::EulerRot::YZX, camera_rotation.x, camera_rotation.y, 0.0) * glam::Vec3::Z;
            let right = glam::Quat::from_axis_angle(glam::Vec3::Y, camera_rotation.x + std::f32::consts::PI / 2.0) * glam::Vec3::Z;

            let mut direction = glam::Vec3::ZERO;

            if keyboard.is_pressed(VirtualKeyCode::W) {
                direction += forward;
            }
            
            if keyboard.is_pressed(VirtualKeyCode::S) {
                direction += -forward;
            }
            
            if keyboard.is_pressed(VirtualKeyCode::A) {
                direction += -right;
            }
            
            if keyboard.is_pressed(VirtualKeyCode::D) {
                direction += right;
            }
            
            if keyboard.is_pressed(VirtualKeyCode::Space) {
                direction += glam::Vec3::Y;
            }
            
            if keyboard.is_pressed(VirtualKeyCode::LControl) {
                direction += glam::Vec3::NEG_Y;
            }

            direction = direction.normalize_or_zero();

            camera_position += direction * speed * FRAME_LENGTH;
            camera_rotation += mouse_delta * angular_speed * FRAME_LENGTH;
        }
        
        // Update camera/matrices
        let model = coordinate_system_matrix * glam::Mat4::from_translation(-centre);
        // println!("Model: {}", model);
        // println!("Camera: {}", camera_position);
        let view = glam::Mat4::from_rotation_translation(glam::Quat::from_euler(glam::EulerRot::YXZ, camera_rotation.x, camera_rotation.y, 0.0), camera_position).inverse();
        let perspective = {
            let (width, height) = target.get_dimensions();
            let aspect = width as f32 / height as f32;
            glam::Mat4::perspective_lh(std::f32::consts::FRAC_PI_3, aspect, 0.1, 10_000.0)
        };

        // Render

        target.clear_color_and_depth((135.0/255.0, 206.0/255.0, 235.0/255.0, 1.0), 1.0);
        target.draw(&vertex_buffer, &indices, &program, &uniform! {
            // mvp: mvp.to_cols_array_2d(),
            u_model: model.to_cols_array_2d(),
            u_view: view.to_cols_array_2d(),
            u_perspective: perspective.to_cols_array_2d(),
            u_coordinate_system: coordinate_system_matrix.to_cols_array_2d(),
        },
                    &glium::DrawParameters {
                        depth: glium::Depth {
                            test: glium::DepthTest::IfLess,
                            write: true,
                            ..Default::default()
                        },
                        // polygon_mode: glium::PolygonMode::Point,
                        // multisampling: true,
                        ..Default::default()
                    }).unwrap();
        target.finish().unwrap();
    });
}
