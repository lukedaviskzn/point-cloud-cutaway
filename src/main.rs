#[macro_use] extern crate glium;
#[macro_use] extern crate maplit;

use std::{sync::mpsc::{self, Receiver}, thread, time::Instant, cell::RefCell, borrow::BorrowMut};

use glium::{glutin::{self, event::{VirtualKeyCode, MouseButton, ElementState}, dpi::PhysicalPosition}, Surface, program::ProgramCreationInput, framebuffer::SimpleFrameBuffer};
use las::{Reader, Read};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use clap::Parser;

use crate::input::{KeyboardManager, MouseManager, MouseButtonState};

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
    file: Option<String>,
    #[clap(short, long, value_parser, about, default_value_t = 0.1)]
    /// Base size of the points, in same units as the file
    point_size: f32,
    #[clap(short, long, value_parser, about, default_value_t = 0)]
    /// Number of points to render, only load first n points. (0 to load all points)
    num_points: u64,
}

#[derive(PartialEq, Eq, Debug)]
enum DrawTool {
    Pencil,
    Eraser,
    RoomIdentification,
}

const FPS: f32 = 60.0;
const FRAME_LENGTH: f32 = 1.0/FPS;
const BATCH_SIZE: u64 = 500_000;

const Z_NEAR: f32 = 0.1;
const Z_FAR: f32 = 1000.0;

const CLEAR_COLOUR: (f32, f32, f32, f32) = (135.0/255.0, 206.0/255.0, 235.0/255.0, 1.0);

fn main() {
    // Profiling
    let server_addr = format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT);
    eprintln!("Serving profile data on {}", server_addr);

    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();

    puffin::set_scopes_on(true);

    // Setup
    let args = Args::parse();
    let filename = args.file;
    let mut point_size = args.point_size;

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Point Cloud Cutaway Renderer");
    let cb = glutin::ContextBuilder::new()
        .with_gl_profile(glutin::GlProfile::Core)
        .with_multisampling(4);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let mut egui_glium = egui_glium::EguiGlium::new(&display, &event_loop);

    {
        let mut fonts = egui::FontDefinitions::default();

        // Install my own font (maybe supporting non-latin characters).
        // .ttf and .otf files supported.
        fonts.font_data.insert(
            "icons".to_owned(),
            egui::FontData::from_static(include_bytes!(
                "../fonts/Font Awesome 6 Free-Solid-900.otf"
            )),
        );

        fonts
            .families
            .entry(egui::FontFamily::Name("icons".into()))
            .or_default()
            .push("icons".to_owned());

        // Tell egui to use these fonts:
        egui_glium.egui_ctx.set_fonts(fonts);
    }

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

    let mut drawing_mode = false;

    let mut active_tool = DrawTool::Pencil;

    // let mut cutaway_file = None;
    // let mut cutaway_slice_file = None;
    // let mut cutaway_slice_processed_file = None;

    let mut cutaway_image: Option<image::ImageBuffer<_, _>> = None;
    let mut cutaway_slice_image: Option<image::ImageBuffer<_, _>> = None;
    let mut cutaway_slice_processed_image: Option<image::ImageBuffer<_, _>> = None;

    // Flip y and z
    let coordinate_system_matrix = glam::mat4(
        glam::vec4(1.0, 0.0, 0.0, 0.0),
        glam::vec4(0.0, 0.0, 1.0, 0.0),
        glam::vec4(0.0, 1.0, 0.0, 0.0),
        glam::vec4(0.0, 0.0, 0.0, 1.0),
    );

    let mut keyboard = KeyboardManager::new();
    let mut mouse = MouseManager::new();

    // let mut shape = vec![];

    let num_points = args.num_points;
    let mut total_points = 0;

    let mut centre = None;
    let mut rx = None;

    // Keeps track of loading progress, -1 = no loading happening right now
    let mut batch_number = -1;

    if let Some(filename) = filename {
        (total_points, centre, rx) = {
            let (n, c, r) = load_point_cloud(&filename, num_points);
            (n, Some(c), Some(r))
        };
        batch_number = 0;
    }

    let mut vertex_buffers = vec![];
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::Points);
    let quad_indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

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

    let drawing_program = {
        let vertex_shader_src = include_str!("shaders/drawing.vert");
        let fragment_shader_src = include_str!("shaders/drawing.frag");
        
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
    
    let mut idle_time = 0.0;

    let mut cutaway_queued = false;

    let mut path_rx: Option<Receiver<String>> = None;

    let fullscreen_quad = glium::VertexBuffer::new(&display, &[
        Vertex {
            position: [-1.0, -1.0, 0.0],
            colour: [0, 0, 0],
        },
        Vertex {
            position: [-1.0, 1.0, 0.0],
            colour: [0, 0, 0],
        },
        Vertex {
            position: [1.0, 1.0, 0.0],
            colour: [0, 0, 0],
        },
        Vertex {
            position: [-1.0, -1.0, 0.0],
            colour: [0, 0, 0],
        },
        Vertex {
            position: [1.0, 1.0, 0.0],
            colour: [0, 0, 0],
        },
        Vertex {
            position: [1.0, -1.0, 0.0],
            colour: [0, 0, 0],
        },
    ]).unwrap();
    
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
                        mouse.update(button, state);
                        
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
                    glutin::event::WindowEvent::CursorMoved { position, .. } => {
                        mouse.update_position(glam::Vec2::new(position.x as f32, position.y as f32));
                        return;
                    }
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
        if !drawing_mode {
            puffin::profile_scope!("update");
            
            if !mouse_locked {
                mouse_delta = glam::Vec2::ZERO;
            }

            // if frame_counter % FPS as u64 == 0 {
            //     println!("{} {:.2}", delta_t.as_millis(), 1.0e9 / (delta_t.as_nanos() as f64));
            // }
            // frame_counter += 1;

            if let Some(r) = &path_rx {
                match r.try_recv() {
                    Ok(path) => {
                        (total_points, centre, rx) = {
                            let (n, c, r) = load_point_cloud(&path, num_points);
                            (n, Some(c), Some(r))
                        };
                        vertex_buffers = vec![];
                        batch_number = 0;
                    },
                    Err(mpsc::TryRecvError::Disconnected) => {
                        path_rx = None;
                    },
                    Err(mpsc::TryRecvError::Empty) => {},
                }
            }

            if let Some(r) = &rx {
                match r.try_recv() {
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
                                // size: point_size,
                            }
                        }).collect();
                        // shape.append(&mut batch);
    
                        vertex_buffers.push(glium::VertexBuffer::new(&display, &batch).unwrap());
    
                        batch_number += 1;

                        println!("Processed Batch {}", batch_number);
                    },
                    Err(mpsc::TryRecvError::Disconnected) => {
                        batch_number = -1;
                        rx = None;
                    },
                    Err(mpsc::TryRecvError::Empty) => {},
                }
            }

            // Handle movement
            
            // speed in units per second
            let speed = if keyboard.is_pressed(VirtualKeyCode::LShift) {
                75.0
            } else {
                15.0
            };
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
                        ui.add(egui::ProgressBar::new(batch_number as f32 / (total_points / BATCH_SIZE + 1) as f32).show_percentage());
                    } else {
                        if ui.add_enabled(path_rx.is_none(), egui::Button::new("Load Point Cloud")).clicked() {
                            let channels = mpsc::channel();
                            path_rx = Some(channels.1);
                            let tx = channels.0;
                            
                            thread::spawn(move || {
                                if let Some(path) = rfd::FileDialog::new().pick_file() {
                                    if let Some(path) = path.to_str() {
                                        tx.send(path.to_owned()).unwrap();
                                    }
                                }
                            });
                        }
    
                        ui.separator();
                        
                        // ui.add(egui::Slider::new(&mut clipping_dist, 0.4..=1.0).logarithmic(true));
                        ui.checkbox(&mut clipping, "Show Cutaway");
                        ui.small("Use W/S keys to control clipping distance.");

                        ui.add(egui::Slider::new(&mut point_size, 0.001..=20.0).logarithmic(true).text("Point Size"));
                        
                        // egui::ComboBox::from_label("Colour Format")
                        // .selected_text(colour_format_options[colour_format as usize])
                        // .show_ui(ui, |ui| {
                        //     for option in colour_format_options.iter().enumerate() {
                        //         ui.selectable_value(&mut colour_format, option.0 as i32, *option.1);
                        //     }
                        // });

                        if ui.button("Render").clicked() {
                            cutaway_queued = true;
                        }
    
                        ui.separator();
    
                        ui.collapsing("Debug", |ui| {
                            ui.checkbox(&mut show_slice, "Show Slice");
                            ui.checkbox(&mut show_outline_plane, "Show Outline Plane");
                        });
                    }

                    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                        ui.label(format!("Idle: {:.2} ms", idle_time * 1000.0));
                        ui.label(format!("FPS: {:.2}", 1.0e9 / (delta_t.as_nanos() as f64)));
                        ui.label(format!("MS: {:.2} ms", delta_t.as_nanos() as f64 / 1.0e6));
                    });
                });
            });
        } else {
            // Unlock mouse
            if mouse_locked {
                let gl_window = display.gl_window();
                let window = gl_window.window();
                
                let _ = window.set_cursor_grab(glutin::window::CursorGrabMode::None);
                let _ = window.set_cursor_visible(true);

                mouse_locked = false;
            }

            egui_glium.run(&display, |egui_ctx| {
                puffin::profile_scope!("update_gui");
                egui::SidePanel::left("my_side_panel").show(egui_ctx, |ui| {
                    let pencil = egui::RichText::new('\u{f303}'.to_string()).family(egui::FontFamily::Name("icons".into()));
                    let eraser = egui::RichText::new('\u{f12d}'.to_string()).family(egui::FontFamily::Name("icons".into()));
                    let room = egui::RichText::new('\u{f015}'.to_string()).family(egui::FontFamily::Name("icons".into()));
                    
                    if ui.button(pencil).clicked() {
                        active_tool = DrawTool::Pencil;
                    }
                    if ui.button(eraser).clicked() {
                        active_tool = DrawTool::Eraser;
                    }
                    if ui.button(room).clicked() {
                        active_tool = DrawTool::RoomIdentification;
                    }

                    ui.label(egui::RichText::new("Room Identification").strong());
                    ui.colored_label(egui::Color32::RED, "Wall and Floor: Red");
                    ui.colored_label(egui::Color32::BLUE, "Room and Exterior: Blue");

                    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                        ui.label(format!("Idle: {:.2} ms", idle_time * 1000.0));
                        ui.label(format!("FPS: {:.2}", 1.0e9 / (delta_t.as_nanos() as f64)));
                        ui.label(format!("MS: {:.2} ms", delta_t.as_nanos() as f64 / 1.0e6));
                    });
                });
            });

            if mouse.is_pressed(MouseButton::Left) || mouse.is_pressed(MouseButton::Right) {
                if let Some(image) = cutaway_slice_processed_image.borrow_mut() {
                    let last_pos = mouse.last_position();
                    let pos = mouse.position();
                    
                    for (lx, ly) in line_drawing::Bresenham::new((last_pos.x as i32, last_pos.y as i32), (pos.x as i32, pos.y as i32)) {
                        match active_tool {
                            DrawTool::Pencil => {
                                image.put_pixel(lx as u32, ly as u32, image::Rgba([0, 0, 0, 255]));
                            },
                            DrawTool::Eraser => {
                                for cy in (ly - 5)..(ly + 5) {
                                    for cx in (lx - 5)..(lx + 5) {
                                        if (cx-lx)*(cx-lx) + (cy-ly)*(cy-ly) <= 5*5 {
                                            image.put_pixel(cx as u32, cy as u32, image::Rgba([255, 255, 255, 0]));
                                        }
                                    }
                                }
                            },
                            DrawTool::RoomIdentification => {
                                let left_pressed = mouse.button_state(MouseButton::Left) == MouseButtonState::JustPressed;
                                let right_pressed = mouse.button_state(MouseButton::Right) == MouseButtonState::JustPressed;

                                if left_pressed || right_pressed {
                                    let target_colour = if left_pressed {
                                        image::Rgba([0, 0, 255, 0])
                                    } else {
                                        image::Rgba([255, 0, 0, 0])
                                    };
                                    
                                    let start_pos = {
                                        let pos = mouse.position();
                                        (pos.x as u32, pos.y as u32)
                                    };
                                    
                                    // Cannot be black or same as target
                                    let start_colour = *image.get_pixel(start_pos.0, start_pos.1);

                                    if start_colour != image::Rgba([0, 0, 0, 255]) && start_colour != target_colour {
                                        let dimensions = image.dimensions();
    
                                        let mut stack = vec![start_pos];
    
                                        while let Some(point) = stack.pop() {
                                            let pixel = *image.get_pixel(point.0, point.1);
    
                                            if pixel != start_colour {
                                                continue;
                                            }
                                            
                                            image.put_pixel(point.0, point.1, target_colour);

                                            if point.0 > 0 {
                                                stack.push((point.0 - 1, point.1));
                                            }
                                            if point.1 > 0 {
                                                stack.push((point.0, point.1 - 1));
                                            }
                                            if point.0 < dimensions.0 - 1 {
                                                stack.push((point.0 + 1, point.1));
                                            }
                                            if point.1 < dimensions.1 - 1 {
                                                stack.push((point.0, point.1 + 1));
                                            }
    
                                            // 1. If node is not Inside return.
                                            // 2. Set the node
                                            // 3. Perform Flood-fill one step to the south of node.
                                            // 4. Perform Flood-fill one step to the north of node
                                            // 5. Perform Flood-fill one step to the west of node
                                            // 6. Perform Flood-fill one step to the east of node
                                            // 7. Return.
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            mouse.on_new_frame();
        }
        
        {
            puffin::profile_scope!("render");
            
            // Update camera/matrices
            let model = coordinate_system_matrix * glam::Mat4::from_translation(-centre.unwrap_or(glam::Vec3::ZERO));
            let view = glam::Mat4::from_rotation_translation(glam::Quat::from_euler(glam::EulerRot::YXZ, camera_rotation.x, camera_rotation.y, 0.0), camera_position).inverse();
            
            // Perspective
            // let projection = {
            //     let (width, height) = target.get_dimensions();
            //     let aspect = width as f32 / height as f32;
            //     glam::Mat4::perspective_lh(FOVY, aspect, 0.1, 10_000.0)
            // };

            let zoom = 2.0_f32.powf(-camera_zoom / 10.0);

            // Orthographic
            let projection = {
                let (width, height) = target.get_dimensions();
                let (width, height) = (width as f32, height as f32);
                let aspect = height / width;
                glam::Mat4::orthographic_lh(-0.5 * zoom, 0.5 * zoom, -aspect * 0.5 * zoom, aspect * 0.5 * zoom, Z_NEAR, Z_FAR)
            };

            let modelview = view * model;

            // Render

            let mut cutaway_texture = None;
            let mut cutaway_slice_texture = None;
            let mut _cutaway_depth = None;
            
            let mut cutaway_buffer: RefCell<Option<SimpleFrameBuffer>> = RefCell::new(None);
            let mut cutaway_slice_buffer: RefCell<Option<SimpleFrameBuffer>> = RefCell::new(None);

            if cutaway_queued {
                cutaway_texture = Some(glium::texture::Texture2d::empty_with_format(&display,
                    glium::texture::UncompressedFloatFormat::U8U8U8U8,
                    glium::texture::MipmapsOption::NoMipmap, window_width, window_height).unwrap());
                cutaway_slice_texture = Some(glium::texture::Texture2d::empty_with_format(&display,
                    glium::texture::UncompressedFloatFormat::U8U8U8U8,
                    glium::texture::MipmapsOption::NoMipmap, window_width, window_height).unwrap());
                _cutaway_depth = Some(glium::framebuffer::DepthRenderBuffer::new(&display, 
                    glium::texture::DepthFormat::F32, window_width, window_height).unwrap());
                
                if let Some(cutaway_texture) = &cutaway_texture {
                    if let Some(cutaway_depth) = &_cutaway_depth {
                        cutaway_buffer = RefCell::new(glium::framebuffer::SimpleFrameBuffer::with_depth_buffer(&display, cutaway_texture, cutaway_depth).ok());
                    }
                }
                if let Some(cutaway_slice_texture) = &cutaway_slice_texture {
                    cutaway_slice_buffer = RefCell::new(glium::framebuffer::SimpleFrameBuffer::new(&display, cutaway_slice_texture).ok());
                }

                cutaway_queued = false;
            }

            {
                puffin::profile_scope!("clear_colour");
                if show_outline_plane {
                    target.clear_color_and_depth((1.0, 1.0, 1.0, 0.0), 1.0);
                } else {
                    target.clear_color_and_depth(CLEAR_COLOUR, 1.0);
                }

                if let Some(cutaway_buffer) = &mut *cutaway_buffer.borrow_mut() {
                    cutaway_buffer.clear_color_and_depth(CLEAR_COLOUR, 1.0);
                }
                if let Some(cutaway_slice_buffer) = &mut *cutaway_slice_buffer.borrow_mut() {
                    cutaway_slice_buffer.clear_color(1.0, 1.0, 1.0, 0.0);
                }
            }
            
            if !drawing_mode {
                puffin::profile_scope!("queue_points");
                for vertex_buffer in &vertex_buffers {
                    let p = if show_outline_plane {
                        &debug_program
                    } else {
                        &program
                    };

                    let uniforms = uniform! {
                        u_modelview: modelview.to_cols_array_2d(),
                        u_projection: projection.to_cols_array_2d(),
                        // u_colour_format: colour_format,
                        // u_clipping_dist: clipping_dist,
                        u_clipping: clipping,
                        u_slice: show_slice,
                        u_slice_width: 0.000025_f32,
                        u_zoom: window_width as f32 / zoom,
                        u_size: point_size,
                    };

                    let draw_params = glium::DrawParameters {
                        depth: glium::Depth {
                            test: glium::DepthTest::IfLess,
                            write: true,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    
                    target.draw(vertex_buffer, &indices, p, &uniforms, &draw_params).unwrap();

                    if let Some(cutaway_buffer) = &mut *cutaway_buffer.borrow_mut() {
                        puffin::profile_scope!("draw_render_frame");
                        cutaway_buffer.draw(vertex_buffer, &indices, &program, &uniforms, &draw_params).unwrap();
                    }
                    if let Some(cutaway_slice_buffer) = &mut *cutaway_slice_buffer.borrow_mut() {
                        puffin::profile_scope!("draw_render_slice");
                        cutaway_slice_buffer.draw(vertex_buffer, &indices, &debug_program, &uniforms, &Default::default()).unwrap();
                    }
                }
            } else {
                let cutaway_texture = {
                    let image = cutaway_image.as_ref().unwrap();
                    let data: Vec<u8> = image.to_vec();
                    let dimensions = image.dimensions();
                    let raw = glium::texture::RawImage2d::from_raw_rgba_reversed(&data, dimensions);

                    glium::texture::Texture2d::new(&display, raw).unwrap()
                };
                let cutaway_slice_texture = {
                    let image = cutaway_slice_processed_image.as_ref().unwrap();
                    let data: Vec<u8> = image.to_vec();
                    let dimensions = image.dimensions();
                    let raw = glium::texture::RawImage2d::from_raw_rgba_reversed(&data, dimensions);

                    glium::texture::Texture2d::new(&display, raw).unwrap()
                };

                target.draw(&fullscreen_quad, &quad_indices, &drawing_program, 
                    &uniform! {
                        u_cutaway: cutaway_texture,
                        u_cutaway_slice: cutaway_slice_texture,
                    }, 
                    &glium::DrawParameters {
                    backface_culling: glium::BackfaceCullingMode::CullingDisabled,
                    ..Default::default()
                }).unwrap();
            }

            {
                puffin::profile_scope!("queue_gui");
                egui_glium.paint(&display, &mut target);
            }
            
            {
                puffin::profile_scope!("finish_frame");
                target.finish().unwrap();
            }

            // Process cutaway
            if let Some(cutaway_texture) = cutaway_texture {
                let cutaway: glium::texture::RawImage2d<_> = cutaway_texture.read();
                let mut image = image::RgbaImage::from_raw(cutaway.width, cutaway.height, (*cutaway.data).to_vec()).unwrap();
                image::imageops::flip_vertical_in_place(&mut image);

                // let dir = tempfile::tempdir().unwrap();

                // eprintln!("Saving cutaway layers to {}", dir.path().display());

                // cutaway_file = Some(dir.path().join("cutaway0.png"));
                // image.save(cutaway_file.as_ref().unwrap()).unwrap();

                cutaway_image = Some(image);
            
                if let Some(cutaway_slice_texture) = cutaway_slice_texture {
                    let cutaway_slice: glium::texture::RawImage2d<_> = cutaway_slice_texture.read();
                    let mut image = image::RgbaImage::from_raw(cutaway_slice.width, cutaway_slice.height, (*cutaway_slice.data).to_vec()).unwrap();
                    image::imageops::flip_vertical_in_place(&mut image);
                    
                    // cutaway_slice_file = Some(dir.path().join("cutaway1_unprocessed.png"));
                    // image.save(cutaway_slice_file.as_ref().unwrap()).unwrap();

                    cutaway_slice_image = Some(image.clone());
                    
                    let mut points = vec![];

                    for (x, y, colour) in image.enumerate_pixels() {
                        if colour.0[3] > 128_u8 {
                            points.push([x as i32, y as i32]);
                        }
                    }

                    let kdtree = kd_tree::KdTree::build(points);

                    for [x, y] in kdtree.iter() {
                        let close_points = kdtree.within_radius(&[*x, *y], (f32::max(point_size * zoom, 1.0) * 10.0) as i32);

                        for close_point in close_points {
                            for (lx, ly) in line_drawing::Bresenham::new((*x, *y), (close_point[0], close_point[1])) {
                                image.put_pixel(lx as u32, ly as u32, image::Rgba([0, 0, 0, 255]));
                            }
                        }
                    }
                    
                    // cutaway_slice_processed_file = Some(dir.path().join("cutaway1.png"));
                    // image.save(cutaway_slice_processed_file.as_ref().unwrap()).unwrap();

                    cutaway_slice_processed_image = Some(image);

                    drawing_mode = true;

                    // image.save("output/cutaway_slice_processed.png").unwrap();
                }
            }
        }
        
        if !drawing_mode {
            puffin::profile_scope!("idle");

            let now = Instant::now();
            let duration_left = next_frame_time - now;

            idle_time = duration_left.as_nanos() as f32 / 1.0e9;

            // wait until next frame
            while now.elapsed() < duration_left {}
        } else {
            idle_time = f32::NAN;
        }
    });
}

fn load_point_cloud(filename: &str, num_points: u64) -> (u64, glam::Vec3, Receiver<Vec<las::Point>>) {
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
    let n = if num_points == 0 {
        total_points
    } else {
        num_points
    };
    
    // let mut i = 0;
    let mut points_processed = 0;

    if n < total_points {
        println!("Loading {} of {} points", n, total_points);
    } else {
        println!("Loading {} points", n);
    }
    
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        puffin::profile_scope!("load_file");
        
        // let mut last_progress = 0;

        let mut batch = vec![];
        let mut batch_number = 0;

        while let Some(Ok(point)) = reader.read() {
            batch.push(point);

            // i += 1;
            points_processed += 1;

            if points_processed % BATCH_SIZE == 0 {
                puffin::profile_scope!("send_batch");
                tx.send(batch).unwrap();
                batch = vec![];
                batch_number += 1;
                println!("Loaded Batch {}/{}", batch_number, n / BATCH_SIZE + 1);
            }

            if points_processed > n {
                tx.send(batch).unwrap();
                batch = vec![];
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

    return (n, centre, rx);
}
