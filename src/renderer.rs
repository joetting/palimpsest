use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

use crate::camera::Camera;
use crate::chunk::{Vertex, CHUNK_SIZE};
use crate::world::{World, ChunkPos};

// ─── Uniform buffer ─────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj:  [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time:       f32,
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj:  glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 3],
            time:       0.0,
        }
    }

    fn update(&mut self, camera: &Camera) {
        self.view_proj  = camera.view_proj().to_cols_array_2d();
        self.camera_pos = camera.position.to_array();
        self.time       += 0.016;
    }
}

// ─── GPU mesh ───────────────────────────────────────────────────────────────

struct GpuMesh {
    vertex_buf: wgpu::Buffer,
    index_buf:  wgpu::Buffer,
    index_count: u32,
}

// ─── Renderer ───────────────────────────────────────────────────────────────

pub struct Renderer {
    pub surface:      wgpu::Surface<'static>,
    pub device:       wgpu::Device,
    pub queue:        wgpu::Queue,
    pub config:       wgpu::SurfaceConfiguration,
    pub size:         winit::dpi::PhysicalSize<u32>,

    pipeline:         wgpu::RenderPipeline,
    uniform_buf:      wgpu::Buffer,
    uniform_bg:       wgpu::BindGroup,
    depth_texture:    wgpu::Texture,
    depth_view:       wgpu::TextureView,
    uniforms:         Uniforms,
    gpu_meshes:       HashMap<ChunkPos, GpuMesh>,
}

impl Renderer {
    pub async fn new(window: &winit::window::Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // SAFETY: we keep `window` alive for the lifetime of the surface
        let surface = unsafe {
            instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(window).unwrap()
            ).unwrap()
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage:        wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:  size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ── Uniforms ──────────────────────────────────────────────────────
        let uniforms = Uniforms::new();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("uniform_bg"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ── Shader & pipeline ─────────────────────────────────────────────
        let shader_src = include_str!("terrain.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("terrain_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("pipeline_layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("terrain_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: "vs_main",
                buffers:     &[Vertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology:          wgpu::PrimitiveTopology::TriangleList,
                front_face:        wgpu::FrontFace::Ccw,
                cull_mode:         Some(wgpu::Face::Back),
                polygon_mode:      wgpu::PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare:       wgpu::CompareFunction::Less,
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview:   None,
        });

        // ── Depth texture ─────────────────────────────────────────────────
        let (depth_texture, depth_view) = make_depth_texture(&device, size.width, size.height);

        Self {
            surface, device, queue, config, size,
            pipeline, uniform_buf, uniform_bg,
            depth_texture, depth_view,
            uniforms,
            gpu_meshes: HashMap::new(),
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.size = new_size;
        self.config.width  = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        let (dt, dv) = make_depth_texture(&self.device, new_size.width, new_size.height);
        self.depth_texture = dt;
        self.depth_view    = dv;
    }

    pub fn render(
        &mut self,
        world:  &World,
        camera: &Camera,
    ) -> Result<(), wgpu::SurfaceError> {

        // ── Update GPU meshes ─────────────────────────────────────────────
        // Upload new/changed meshes
        for (pos, mesh) in &world.meshes {
            if self.gpu_meshes.contains_key(pos) { continue; }
            if mesh.vertices.is_empty() { continue; }

            let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("vb"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage:    wgpu::BufferUsages::VERTEX,
            });
            let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some("ib"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage:    wgpu::BufferUsages::INDEX,
            });
            self.gpu_meshes.insert(*pos, GpuMesh {
                vertex_buf:  vb,
                index_buf:   ib,
                index_count: mesh.indices.len() as u32,
            });
        }

        // Remove GPU meshes for unloaded chunks
        self.gpu_meshes.retain(|pos, _| world.meshes.contains_key(pos));

        // ── Update uniforms ───────────────────────────────────────────────
        self.uniforms.update(camera);
        self.queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        // ── Render ────────────────────────────────────────────────────────
        let output = self.surface.get_current_texture()?;
        let view   = output.texture.create_view(&Default::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.68, g: 0.60, b: 0.50, a: 1.0  // warm ochre sky
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view:        &self.depth_view,
                    depth_ops:   Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes:   None,
                occlusion_query_set: None,
            });

            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.uniform_bg, &[]);

            let cs = CHUNK_SIZE as f32;
            let view_proj = camera.view_proj();
            let cam = camera.position;

            for (pos, mesh) in &self.gpu_meshes {
                // Very rough frustum cull - skip obviously behind or far chunks
                let chunk_center = glam::Vec3::new(
                    pos.0 as f32 * cs + cs * 0.5,
                    pos.1 as f32 * cs + cs * 0.5,
                    pos.2 as f32 * cs + cs * 0.5,
                );
                let dist = (chunk_center - cam).length();
                if dist > 230.0 { continue; }

                rp.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                rp.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn make_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("depth"),
        size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Depth32Float,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats:    &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}
