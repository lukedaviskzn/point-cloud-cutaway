use std::collections::HashMap;

use glium::glutin::event::{VirtualKeyCode, KeyboardInput, ElementState, MouseButton};

pub struct KeyboardManager {
    state: HashMap<VirtualKeyCode, bool>,
}

impl KeyboardManager {
    pub fn new() -> KeyboardManager {
        KeyboardManager{
            state: hashmap!{},
        }
    }

    pub fn update(&mut self, event: KeyboardInput) {
        if let Some(key) = event.virtual_keycode {
            self.state.insert(key, event.state == ElementState::Pressed);
        }
    }

    pub fn is_pressed(&self, key: VirtualKeyCode) -> bool {
        return *self.state.get(&key).unwrap_or(&false);
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum MouseButtonState {
    Pressed,
    Released,
    JustPressed,
    JustReleased,
}

pub struct MouseManager {
    state: HashMap<MouseButton, MouseButtonState>,
    position: glam::Vec2,
    last_position: glam::Vec2,
    new_frame: bool,
}

impl MouseManager {
    pub fn new() -> MouseManager {
        MouseManager {
            state: hashmap!{},
            position: glam::Vec2::NAN,
            last_position: glam::Vec2::NAN,
            new_frame: true,
        }
    }

    pub fn update(&mut self, button: MouseButton, state: ElementState) {
        self.state.insert(button, match state {
            ElementState::Pressed => MouseButtonState::JustPressed,
            ElementState::Released => MouseButtonState::JustReleased,
        });
    }

    pub fn update_position(&mut self, position: glam::Vec2) {
        if self.new_frame {
            self.last_position = self.position;
            self.new_frame = false;
        }
        self.position = position;
    }

    pub fn on_new_frame(&mut self) {
        self.new_frame = true;
        for (_, val) in self.state.iter_mut() {
            match val {
                MouseButtonState::JustPressed => *val = MouseButtonState::Pressed,
                MouseButtonState::JustReleased => *val = MouseButtonState::Released,
                _ => {},
            }
        }
    }

    pub fn is_pressed(&self, button: MouseButton) -> bool {
        return match *self.state.get(&button).unwrap_or(&MouseButtonState::Released) {
            MouseButtonState::JustPressed => true,
            MouseButtonState::Pressed => true,
            MouseButtonState::JustReleased => false,
            MouseButtonState::Released => false,
        };
    }

    pub fn button_state(&self, button: MouseButton) -> MouseButtonState {
        return *self.state.get(&button).unwrap_or(&MouseButtonState::Released);
    }

    pub fn last_position(&self) -> glam::Vec2 {
        return self.last_position;
    }

    pub fn position(&self) -> glam::Vec2 {
        return self.position;
    }
}
