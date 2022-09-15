#[macro_use] extern crate glium;
#[macro_use] extern crate maplit;

use std::{sync::mpsc, thread};

use glium::{glutin::{self, event::{VirtualKeyCode, MouseButton, ElementState}, dpi::PhysicalPosition}, Surface, program::ProgramCreationInput};
use kd_tree::{KdTree, KdPoint};
use las::{Reader, Read};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use clap::Parser;

use crate::input::KeyboardManager;

mod input;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    colour: [u16; 3],
    size: f32,
}

impl KdPoint for Vertex {
    type Scalar = f32;
    type Dim = typenum::U3;

    fn at(&self, k: usize) -> f32 {
        self.position[k]
    }
}

#[derive(Parser, Debug)]
#[clap(author="Luke Davis", version, about="Renders point cloud information and generated cutaway given specific clipping distance.")]
struct Args {
    #[clap(short, long, value_parser, about)]
    /// Point cloud file path
    file: String,
    #[clap(short, long, value_parser, about, default_value_t = 10.0)]
    /// Base size of the points, in same units as the file
    point_size: f32,
    #[clap(short, long, value_parser, about, default_value_t = 0)]
    /// Number of points to render, only load first n points. (0 to load all points)
    num_points: u64,
}

const FPS: f32 = 60.0;
const FRAME_LENGTH: f32 = 1.0/FPS;
const FOVY: f32 = std::f32::consts::FRAC_PI_3;
const BATCH_SIZE: u64 = 500_000;

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

    let mut clipping_dist = 0.0_f32;
    let mut show_slice = false;

    // Flip y and z
    let coordinate_system_matrix = glam::mat4(
        glam::vec4(1.0, 0.0, 0.0, 0.0),
        glam::vec4(0.0, 0.0, 1.0, 0.0),
        glam::vec4(0.0, 1.0, 0.0, 0.0),
        glam::vec4(0.0, 0.0, 0.0, 1.0),
    );

    let mut keyboard = KeyboardManager::new();

    let mut shape = vec![];

    let (tx, rx) = mpsc::channel();

    let mut reader = Reader::from_path(filename).unwrap();

    let mut colour_format = if reader.header().point_format().has_color {
        2
    } else {
        0
    };
    
    let total_points = reader.header().number_of_points();
    let n = if args.num_points == 0 {
        total_points
    } else {
        args.num_points
    };
    
    thread::spawn(move || {
        let mut i = 0;
        let mut points_processed = 0;

        println!("Loading {} of {} points", n, total_points);

        let mut last_progress = 0;

        let mut batch = vec![];

        while let Some(Ok(point)) = reader.read() {
            // if i < (points_processed) * total_points / n {
            //     i = (points_processed) * total_points / n;
            //     println!("Seeking to {}, {}", (points_processed) * total_points / n, i);
            //     reader.seek(i).unwrap(); // seek is incrediblity slow, slower than just reading and discarding the points
            //     continue;
            // }

            // let colour = if let Some(colour) = point.color {
            //     [colour.red as f32, colour.green as f32, colour.blue as f32]
            // } else {
            //     [1.0, 1.0, 1.0]
            // };

            // let v = Vertex {
            //     position: [point.x as f32, point.y as f32, point.z as f32],
            //     colour: colour,
            //     size: default_point_size,
            // };

            batch.push(point);

            i += 1;
            points_processed += 1;

            if points_processed % BATCH_SIZE == 0 {
                tx.send(batch).unwrap();
                batch = vec![];
            }

            if points_processed > n {
                tx.send(batch).unwrap();
                break;
            }

            let progress = (100 * points_processed) / n; // percentage
            if progress != last_progress {
                last_progress = progress;
                println!("Loading... {}%", progress);
            }
        }

        println!("{} {}", points_processed, i);
        println!("Points Loaded");
    });

    // Wait for points
    for _ in 0..(n / BATCH_SIZE + 1) {
        if let Ok(mut batch) = rx.recv() {
            shape.append(&mut batch);
        } else {
            break;
        }
    }

    // Parse points
    let shape: Vec<_> = shape.par_iter().map(|point| {
        let colour = if let Some(colour) = point.color {
            [colour.red, colour.green, colour.blue]
        } else {
            [u16::MAX; 3]
        };
        
        Vertex {
            position: [point.x as f32, point.y as f32, point.z as f32],
            colour: colour,
            size: default_point_size,
        }
    }).collect();

    // Calculating totals
    let centre = shape.par_iter()
        .map(|x| x.position)
        .reduce(
            || [0.0, 0.0, 0.0],
            |x, y| [x[0]+y[0], x[1]+y[1], x[2]+y[2]],
        );
    
    // Calculate centre
    let centre = glam::vec3(centre[0], centre[1], centre[2]) / n as f32;

    // KdTree
    let shape: Vec<_> = {
        print!("Building KdTree... ");
        let kdtree = KdTree::build_by_ordered_float(shape);
        println!("done");
        print!("Calculating Point Sizes... ");
        kdtree.par_iter().map(|p| {
            let nearests = kdtree.nearests(&p.position, 6);

            let mut max_sdist = 0.0;
            for v in nearests {
                if v.squared_distance > max_sdist {
                    max_sdist = v.squared_distance * 1.5;
                }
            }

            Vertex {
                position: p.position,
                colour: p.colour,
                size: max_sdist.sqrt(),
            }
        }).collect()
    };
    println!("done");

    let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
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

                    if input.state == ElementState::Pressed {
                        if let Some(key) = input.virtual_keycode {
                            match key {
                                VirtualKeyCode::Escape => {
                                    let gl_window = display.gl_window();
                                    let window = gl_window.window();
                                    
                                    let _ = window.set_cursor_grab(glutin::window::CursorGrabMode::None);
                                    let _ = window.set_cursor_visible(true);
        
                                    mouse_locked = false;
                                },
                                VirtualKeyCode::F => {
                                    if colour_format == 1 {
                                        colour_format = 2;
                                    } else if colour_format == 2 {
                                        colour_format = 1;
                                    }

                                    println!("Colour Format: {}", colour_format * 8);
                                },
                                VirtualKeyCode::T => {
                                    show_slice = !show_slice;
                                },
                                _ => {},
                            }
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
        let (window_width, window_height) = target.get_dimensions();

        // Handle Update
        {
            // Get any newly available points
            // let mut changed = false;
            // while let Ok(point) = rx.try_recv() {
            //     shape.push(point);
            //     changed = true;
            // }
            // if changed {
            //     vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
            // }

            let clip_speed = 0.25;
            if keyboard.is_pressed(VirtualKeyCode::Up) {
                clipping_dist += clip_speed;
                if clipping_dist > 100.0 {
                    clipping_dist = 100.0;
                }
            }
            if keyboard.is_pressed(VirtualKeyCode::Down) {
                clipping_dist -= clip_speed;
                if clipping_dist < 0.0 {
                    clipping_dist = 0.0;
                }
            }

            let window_centre = glam::vec2(window_width as f32 / 2.0, window_height as f32 / 2.0);
            
            let mouse_delta = if !mouse_position.is_nan() && mouse_locked {
                let _ = display.gl_window().window().set_cursor_position(PhysicalPosition::new(window_width / 2, window_height / 2));
                mouse_position - window_centre
            } else {
                glam::Vec2::ZERO
            };

            let speed = 5.0; // units per second
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
        let projection = {
            let (width, height) = target.get_dimensions();
            let aspect = width as f32 / height as f32;
            glam::Mat4::perspective_lh(FOVY, aspect, 0.1, 10_000.0)
        };

        // let projection = {
        //     let (width, height) = target.get_dimensions();
        //     let (width, height) = (width as f32, height as f32);
        //     glam::Mat4::orthographic_lh(-width/2.0, width/2.0, -height/2.0, height/2.0, 0.1, 10_000.0)
        // };

        let modelview = view * model;

        // Render

        target.clear_color_and_depth((135.0/255.0, 206.0/255.0, 235.0/255.0, 1.0), 1.0);
        target.draw(&vertex_buffer, &indices, &program, &uniform! {
            u_modelview: modelview.to_cols_array_2d(),
            u_projection: projection.to_cols_array_2d(),
            u_colour_format: colour_format,
            u_clipping_dist: clipping_dist,
            u_slice: show_slice,
            u_window_height: window_height,
            u_fovy: FOVY,
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
