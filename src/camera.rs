use glam::{Mat4, Vec3};
use crate::input::InputState;

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,   // radians, horizontal
    pub pitch: f32, // radians, vertical

    pub move_speed: f32,
    pub mouse_sensitivity: f32,

    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: -0.3,
            move_speed: 20.0,
            mouse_sensitivity: 0.002,
            fov_y: 70_f32.to_radians(),
            aspect: 1280.0 / 720.0,
            near: 0.1,
            far: 1000.0,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    pub fn update(&mut self, input: &InputState, dt: f32) {
        let fwd = self.forward();
        let right = self.right();

        let mut velocity = Vec3::ZERO;
        if input.forward  { velocity += fwd; }
        if input.backward { velocity -= fwd; }
        if input.right    { velocity += right; }
        if input.left     { velocity -= right; }
        if input.up       { velocity += Vec3::Y; }
        if input.down     { velocity -= Vec3::Y; }

        if velocity.length_squared() > 0.0 {
            self.position += velocity.normalize() * self.move_speed * dt;
        }
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw   += dx * self.mouse_sensitivity;
        self.pitch -= dy * self.mouse_sensitivity;
        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
             std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
}
