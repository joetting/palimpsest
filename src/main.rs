mod camera;
mod chunk;
mod renderer;
mod world;
mod input;

use winit::{
    event::{Event, WindowEvent, DeviceEvent, KeyEvent, ElementState, MouseButton},
    event_loop::{EventLoop, ControlFlow},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};
use std::time::Instant;

use camera::Camera;
use input::InputState;
use renderer::Renderer;
use world::World;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Unconformity")
        .with_inner_size(winit::dpi::PhysicalSize::new(1280u32, 720u32))
        .build(&event_loop)
        .unwrap();

    // Grab cursor for FPS-style mouse look
    window.set_cursor_visible(false);
    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);

    let mut renderer = pollster::block_on(Renderer::new(&window));
    let mut world = World::new();
    let mut camera = Camera::new(glam::Vec3::new(128.0, 80.0, 128.0));
    let mut input = InputState::default();
    let mut last_frame = Instant::now();
    let mut focused = true;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),

                WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state, .. }, .. } => {
                    if let PhysicalKey::Code(code) = physical_key {
                        let pressed = state == ElementState::Pressed;
                        match code {
                            KeyCode::KeyW | KeyCode::ArrowUp    => input.forward  = pressed,
                            KeyCode::KeyS | KeyCode::ArrowDown  => input.backward = pressed,
                            KeyCode::KeyA | KeyCode::ArrowLeft  => input.left     = pressed,
                            KeyCode::KeyD | KeyCode::ArrowRight => input.right    = pressed,
                            KeyCode::Space                       => input.up       = pressed,
                            KeyCode::ShiftLeft                  => input.down     = pressed,
                            KeyCode::Escape if pressed          => elwt.exit(),
                            _ => {}
                        }
                    }
                }

                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    if state == ElementState::Pressed && focused {
                        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
                        window.set_cursor_visible(false);
                    }
                }

                WindowEvent::Focused(f) => {
                    focused = f;
                    if !f {
                        let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                        window.set_cursor_visible(true);
                        input = InputState::default();
                    }
                }

                WindowEvent::Resized(size) => {
                    renderer.resize(size);
                }

                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = (now - last_frame).as_secs_f32();
                    last_frame = now;

                    camera.update(&input, dt);
                    world.update(&camera);

                    match renderer.render(&world, &camera) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            renderer.resize(window.inner_size());
                        }
                        Err(e) => eprintln!("Render error: {e:?}"),
                    }
                }

                _ => {}
            },

            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                if focused {
                    camera.rotate(delta.0 as f32, delta.1 as f32);
                }
            }

            Event::AboutToWait => {
                window.request_redraw();
            }

            _ => {}
        }
    }).unwrap();
}
