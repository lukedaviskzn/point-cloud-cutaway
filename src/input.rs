use std::collections::HashMap;

use glium::glutin::event::{VirtualKeyCode, KeyboardInput, ElementState};

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
