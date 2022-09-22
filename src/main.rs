#[macro_use] extern crate glium;
#[macro_use] extern crate maplit;

use std::{sync::mpsc, thread, time::Instant};

use glium::{glutin::{self, event::{VirtualKeyCode, MouseButton, ElementState}, dpi::PhysicalPosition}, Surface, program::ProgramCreationInput};
use las::{Reader, Read};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use clap::Parser;

use crate::input::KeyboardManager;

mod input;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    colour: [u8; 3],
}

#[derive(Parser, Debug)]
#[clap(author="Luke Davis", version, about="Renders point cloud information and generated cutaway given specific clipping distance.")]
struct Args {
    #[clap(short, long, value_parser, about)]
    /// Point cloud file path
    file: String,
    #[clap(short, long, value_parser, about, default_value_t = 0.1)]
    /// Base size of the points, in same units as the file
    point_size: f32,
    #[clap(short, long, value_parser, about, default_value_t = 0)]
    /// Number of points to render, only load first n points. (0 to load all points)
    num_points: u64,
}

const FPS: f32 = 60.0;
const FRAME_LENGTH: f32 = 1.0/FPS;
const BATCH_SIZE: u64 = 500_000;

const Z_NEAR: f32 = 0.1;
const Z_FAR: f32 = 1000.0;

fn main() {
    // Profiling
    let server_addr = format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT);
    eprintln!("Serving profile data on {}", server_addr);

    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();

    puffin::set_scopes_on(true);

    // Setup
    let args = Args::parse();
    let filename = args.file;
    let default_point_size = args.point_size;

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Point Cloud Cutaway Renderer");
    let cb = glutin::ContextBuilder::new()
        .with_gl_profile(glutin::GlProfile::Core)
        .with_multisampling(4);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let mut egui_glium = egui_glium::EguiGlium::new(&display, &event_loop);

    implement_vertex!(Vertex, position, colour/*, size*/);

    let mut camera_position: glam::Vec3 = glam::Vec3::ZERO;
    let mut camera_rotation: glam::Vec2 = glam::vec2(0.0, std::f32::consts::FRAC_PI_2);
    let mut camera_zoom: f32 = -64.0;

    // let mut mouse_position = glam::Vec2::NAN;
    let mut mouse_delta = glam::Vec2::ZERO;

    let mut mouse_locked = false;

    // let mut clipping_dist = 0.0_f32;
    let mut clipping = false;
    let mut show_slice = false;
    let mut show_outline_plane = false;

    // Flip y and z
    let coordinate_system_matrix = glam::mat4(
        glam::vec4(1.0, 0.0, 0.0, 0.0),
        glam::vec4(0.0, 0.0, 1.0, 0.0),
        glam::vec4(0.0, 1.0, 0.0, 0.0),
        glam::vec4(0.0, 0.0, 0.0, 1.0),
    );

    let mut keyboard = KeyboardManager::new();

    // let mut shape = vec![];

    let (tx, rx) = mpsc::channel();

    let mut reader = Reader::from_path(filename).unwrap();

    // let colour_format_options = ["Solid White", "8-Bit Colour", "16-Bit Colour"];
    // let mut colour_format: i32 = if reader.header().point_format().has_color {
    //     2
    // } else {
    //     0
    // };
    
    let centre = {
        let bounds = reader.header().bounds();

        glam::vec3(
            (bounds.min.x + bounds.max.x) as f32 / 2.0,
            (bounds.min.y + bounds.max.y) as f32 / 2.0,
            (bounds.min.z + bounds.max.z) as f32 / 2.0,
        )
    };
    
    let total_points = reader.header().number_of_points();
    let n = if args.num_points == 0 {
        total_points
    } else {
        args.num_points
    };
    
    thread::spawn(move || {
        puffin::profile_scope!("load_file");
        // let mut i = 0;
        let mut points_processed = 0;

        println!("Loading {} of {} points", n, total_points);

        // let mut last_progress = 0;

        let mut batch = vec![];

        while let Some(Ok(point)) = reader.read() {
            batch.push(point);

            // i += 1;
            points_processed += 1;

            if points_processed % BATCH_SIZE == 0 {
                puffin::profile_scope!("send_batch");
                tx.send(batch).unwrap();
                batch = vec![];
            }

            if points_processed > n {
                tx.send(batch).unwrap();
                break;
            }
            // let progress = (100 * points_processed) / n; // percentage
            // if progress != last_progress {
            //     last_progress = progress;
            //     println!("Loading... {}%", progress);
            // }
        }

        println!("Points Loaded");
    });

    // KdTree
    // let shape: Vec<_> = {
    //     print!("Building KdTree... ");
    //     let kdtree = KdTree::build_by_ordered_float(shape);
    //     println!("done");
    //     print!("Calculating Point Sizes... ");
    //     kdtree.par_iter().map(|p| {
    //         let nearests = kdtree.nearests(&p.position, 6);

    //         let mut max_sdist = 0.0;
    //         for v in nearests {
    //             if v.squared_distance > max_sdist {
    //                 max_sdist = v.squared_distance * 1.5;
    //             }
    //         }

    //         Vertex {
    //             position: p.position,
    //             colour: p.colour,
    //             size: max_sdist.sqrt(),
    //         }
    //     }).collect()
    // };
    // println!("done");

    let mut vertex_buffers = vec![];
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::Points);

    let program = {
        let vertex_shader_src = include_str!("shaders/main.vert");
        let fragment_shader_src = include_str!("shaders/main.frag");
        
        glium::Program::new(&display, ProgramCreationInput::SourceCode {
            vertex_shader: vertex_shader_src,
            fragment_shader: fragment_shader_src,
            uses_point_size: true,
            tessellation_control_shader: None,
            tessellation_evaluation_shader: None,
            geometry_shader: None,
            transform_feedback_varyings: None,
            outputs_srgb: true,
        }).unwrap()
    };

    let debug_program = {
        let vertex_shader_src = include_str!("shaders/single_pixel.vert");
        let fragment_shader_src = include_str!("shaders/single_pixel.frag");
        
        glium::Program::new(&display, ProgramCreationInput::SourceCode {
            vertex_shader: vertex_shader_src,
            fragment_shader: fragment_shader_src,
            uses_point_size: true,
            tessellation_control_shader: None,
            tessellation_evaluation_shader: None,
            geometry_shader: None,
            transform_feedback_varyings: None,
            outputs_srgb: true,
        }).unwrap()
    };

    let mut last_time = Instant::now();

    let mut _frame_counter = 0_u64;
    
    let mut batch_number = 0;

    let mut idle_time = 0.0;

    let mut cutaway_queued = false;
    
    event_loop.run(move |event, _, control_flow| {

        puffin::profile_function!();

        let next_frame_time = std::time::Instant::now() +
            std::time::Duration::from_nanos((FRAME_LENGTH * 1.0e9) as u64);
        // *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
        // *control_flow = glutin::event_loop::ControlFlow::Poll;

        match event {
            glutin::event::Event::WindowEvent { event, .. } => {
                
                if egui_glium.on_event(&event) {
                    return;
                }
                
                match event {
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
                                    // VirtualKeyCode::F => {
                                    //     if colour_format == 1 {
                                    //         colour_format = 2;
                                    //     } else if colour_format == 2 {
                                    //         colour_format = 1;
                                    //     }

                                    //     println!("Colour Format: {}", colour_format * 8);
                                    // },
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
                        if state == ElementState::Pressed {
                            match button {
                                MouseButton::Left => {
                                    let gl_window = display.gl_window();
                                    let window = gl_window.window();
                                    
                                    if let Err(_) = window.set_cursor_grab(glutin::window::CursorGrabMode::Locked) {
                                        // eprintln!("Failed to lock cursor, confining to window instead! {:?}", err);
                                        if let Err(err) = window.set_cursor_grab(glutin::window::CursorGrabMode::Confined) {
                                            eprintln!("Failed to lock or confine cursor! {:?}", err);
                                            return;
                                        }
                                    }
                                    window.set_cursor_visible(false);

                                    mouse_locked = true;
                                },
                                MouseButton::Right => {
                                    let gl_window = display.gl_window();
                                    let window = gl_window.window();
                                    
                                    let _ = window.set_cursor_grab(glutin::window::CursorGrabMode::None);
                                    let _ = window.set_cursor_visible(true);
        
                                    mouse_locked = false;
                                },
                                _ => {},
                            }
                        }
                        return;
                    },
                    glutin::event::WindowEvent::MouseWheel { delta, .. } => {
                        match delta {
                            glutin::event::MouseScrollDelta::LineDelta(_x, y) => {
                                camera_zoom += y;
                            },
                            _ => {},
                        };
                        return;
                    },
                    _ => return,
                };
            },
            glutin::event::Event::DeviceEvent { event, .. } => match event {
                glutin::event::DeviceEvent::MouseMotion { delta } => {
                    mouse_delta += glam::vec2(delta.0 as f32, delta.1 as f32);
                    return;
                },
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                glutin::event::StartCause::Poll => (),
                _ => return,
            },
            // glutin::event::Event::MainEventsCleared => {
            //     display.gl_window().window().request_redraw();
            // }
            _ => return,
        }

        puffin::GlobalProfiler::lock().new_frame(); // call once per frame!
        
        let mut target = display.draw();
        let (window_width, window_height) = target.get_dimensions();

        let now = Instant::now();
        let delta_t = now - last_time;
        last_time = now;
        
        // Handle Update
        {
            puffin::profile_scope!("update");
            
            if !mouse_locked {
                mouse_delta = glam::Vec2::ZERO;
            }

            // if frame_counter % FPS as u64 == 0 {
            //     println!("{} {:.2}", delta_t.as_millis(), 1.0e9 / (delta_t.as_nanos() as f64));
            // }
            // frame_counter += 1;

            match rx.recv() {
                Ok(batch) => {
                    let batch: Vec<_> = batch.par_iter().map(|point| {
                        let colour = if let Some(colour) = point.color {
                            [(colour.red / 256) as u8, (colour.green / 256) as u8, (colour.blue / 256) as u8]
                        } else {
                            [u8::MAX; 3]
                        };
                        
                        Vertex {
                            position: [point.x as f32, point.y as f32, point.z as f32],
                            colour: colour,
                            // size: default_point_size,
                        }
                    }).collect();
                    // shape.append(&mut batch);

                    vertex_buffers.push(glium::VertexBuffer::new(&display, &batch).unwrap());

                    batch_number += 1;
                },
                Err(_) => {
                    batch_number = -1;
                },
            }

            // Handle movement
            let speed = 15.0; // units per second
            let angular_speed = 0.1; // radians per second (multiplied by mouse speed, equivalent to minimum mouse speed of 1px/frame)
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

            camera_rotation.y = camera_rotation.y.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);

            mouse_delta = glam::Vec2::ZERO;

            if mouse_locked {
                let _ = display.gl_window().window().set_cursor_position(PhysicalPosition::new(window_width / 2, window_height / 2));
            }
        
            egui_glium.run(&display, |egui_ctx| {
                puffin::profile_scope!("update_gui");
                egui::SidePanel::left("my_side_panel").show(egui_ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading(egui::RichText::new("Point Cloud Cutaway Renderer").strong());
                    });

                    ui.separator();

                    if batch_number >= 0 {
                        ui.label("Loading Point Cloud File");
                        ui.add(egui::ProgressBar::new(batch_number as f32 / (n / BATCH_SIZE + 1) as f32).show_percentage());
                    } else {
                        // ui.add(egui::Slider::new(&mut clipping_dist, 0.4..=1.0).logarithmic(true));
                        ui.checkbox(&mut clipping, "Show Cutaway");
                        ui.small("Use W/S keys to control clipping distance.");
                        
                        // egui::ComboBox::from_label("Colour Format")
                        // .selected_text(colour_format_options[colour_format as usize])
                        // .show_ui(ui, |ui| {
                        //     for option in colour_format_options.iter().enumerate() {
                        //         ui.selectable_value(&mut colour_format, option.0 as i32, *option.1);
                        //     }
                        // });
    
                        ui.separator();
    
                        ui.collapsing("Debug", |ui| {
                            ui.checkbox(&mut show_slice, "Show Slice");
                            ui.checkbox(&mut show_outline_plane, "Show Outline Plane");
                        });
                    }

                    if ui.button("Render").clicked() {
                        cutaway_queued = true;
                    }

                    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                        ui.label(format!("MS: {} ms", delta_t.as_millis()));
                        ui.label(format!("FPS: {:.2}", 1.0e9 / (delta_t.as_nanos() as f64)));
                        ui.label(format!("Idle: {:.2} ms", idle_time * 1000.0));
                    });
                });
            });
        }
        
        {
            puffin::profile_scope!("render");
            
            // Update camera/matrices
            let model = coordinate_system_matrix * glam::Mat4::from_translation(-centre);
            // println!("Model: {}", model);
            // println!("Camera: {}", camera_position);
            let view = glam::Mat4::from_rotation_translation(glam::Quat::from_euler(glam::EulerRot::YXZ, camera_rotation.x, camera_rotation.y, 0.0), camera_position).inverse();
            // let projection = {
            //     let (width, height) = target.get_dimensions();
            //     let aspect = width as f32 / height as f32;
            //     glam::Mat4::perspective_lh(FOVY, aspect, 0.1, 10_000.0)
            // };

            let zoom = 2.0_f32.powf(-camera_zoom / 10.0);

            let projection = {
                let (width, height) = target.get_dimensions();
                let (width, height) = (width as f32, height as f32);
                let aspect = height / width;
                glam::Mat4::orthographic_lh(-0.5 * zoom, 0.5 * zoom, -aspect * 0.5 * zoom, aspect * 0.5 * zoom, Z_NEAR, Z_FAR)
            };

            let modelview = view * model;

            // Render

            let mut target = glium::framebuffer::SimpleFrameBuffer;

            {
                puffin::profile_scope!("clear_colour");
                if show_outline_plane {
                    target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);
                } else {
                    target.clear_color_and_depth((135.0/255.0, 206.0/255.0, 235.0/255.0, 1.0), 1.0);
                }
            }
            
            {
                puffin::profile_scope!("queue_points");
                for vertex_buffer in &vertex_buffers {
                    target.draw(vertex_buffer, &indices, if show_outline_plane {&debug_program} else {&program}, 
                        &uniform! {
                            u_modelview: modelview.to_cols_array_2d(),
                            u_projection: projection.to_cols_array_2d(),
                            // u_colour_format: colour_format,
                            // u_clipping_dist: clipping_dist,
                            u_clipping: clipping,
                            u_slice: show_slice,
                            u_slice_width: 0.000025_f32,
                            u_zoom: window_width as f32 / zoom,
                            u_size: default_point_size,
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
                        }
                    ).unwrap();
                }
            }
            

            // {
            //     puffin::profile_scope!("Render Slice Buffer");
            //     for vertex_buffer in &vertex_buffers {
            //         framebuffer.draw(vertex_buffer, &indices, &debug_program, 
            //             &uniform! {
            //                 u_modelview: modelview.to_cols_array_2d(),
            //                 u_projection: projection.to_cols_array_2d(),
            //                 u_slice_width: 0.000025_f32,
            //                 u_zoom: window_width as f32 / zoom,
            //             },
            //             &glium::DrawParameters {
            //                 depth: glium::Depth::default(),
            //                 // polygon_mode: glium::PolygonMode::Point,
            //                 // multisampling: true,
            //                 ..Default::default()
            //             }
            //         ).unwrap();
            //     }
            // }

            // if show_outline_plane {
            //     puffin::profile_scope!("Copy Framebuffer");
            //     target.blit_buffers_from_simple_framebuffer(
            //         &framebuffer,
            //         &Rect { left: 0, bottom: 0, width: window_width, height: window_height },
            //         &BlitTarget { left: 0, bottom: 0, width: window_width as i32, height: window_height as i32 },
            //         glium::uniforms::MagnifySamplerFilter::Linear, BlitMask::color());
            // }

            {
                puffin::profile_scope!("queue_gui");
                egui_glium.paint(&display, &mut target);
            }
            
            {
                puffin::profile_scope!("finish_frame");
                target.finish().unwrap();
            }
        }
        
        {
            puffin::profile_scope!("idle");

            let now = Instant::now();
            let duration_left = next_frame_time - now;

            idle_time = duration_left.as_nanos() as f32 / 1.0e9;

            // wait until next frame
            while now.elapsed() < duration_left {}
        }
    });
}
