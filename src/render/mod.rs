//! Renderer: minimal wgpu-based rendering with a fixed memory footprint.
//!
//! The renderer owns no GPU resources with dynamic lifetime; it holds pipelines,
//! a vertex buffer, and surface configuration. Device and queue are passed in
//! per call, which keeps borrowing simple.

use wgpu::util::DeviceExt;

/// Clear color used each frame (RGBA, 0..=1).
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.1,
    g: 0.1,
    b: 0.12,
    a: 1.0,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

const QUAD_VERTICES: [Vertex; 4] = [
    Vertex { position: [-1.0, -1.0], color: [0.2, 0.6, 1.0, 1.0] },
    Vertex { position: [1.0, -1.0], color: [0.2, 0.6, 1.0, 1.0] },
    Vertex { position: [-1.0, 1.0], color: [0.2, 0.6, 1.0, 1.0] },
    Vertex { position: [1.0, 1.0], color: [0.2, 0.6, 1.0, 1.0] },
];

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];

/// Shader source (inline WGSL, no asset files needed for the engine core).
const VS_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}
"#;

const FS_SRC: &str = r#"
@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
"#;

pub struct Renderer {
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    vsync: bool,
    pipeline: Option<wgpu::RenderPipeline>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
}

impl Renderer {
    pub fn new(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        surface: &wgpu::Surface,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Self {
        let size = winit::dpi::PhysicalSize::new(width, height);
        let surface_capabilities = surface.get_capabilities(adapter);
        let format = surface_capabilities
            .formats
            .first()
            .copied()
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: if vsync {
                wgpu::PresentMode::Fifo
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(device, &config);

        let fs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine-fs-shader"),
            source: wgpu::ShaderSource::Wgsl(FS_SRC.into()),
        });
        let vs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine-vs-shader"),
            source: wgpu::ShaderSource::Wgsl(VS_SRC.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine-pipeline-layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("engine-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine-quad-vertex-buffer"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine-quad-index-buffer"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            config,
            size,
            vsync,
            pipeline: Some(pipeline),
            vertex_buffer: Some(vertex_buffer),
            index_buffer: Some(index_buffer),
            index_count: QUAD_INDICES.len() as u32,
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        surface: &wgpu::Surface,
        width: u32,
        height: u32,
    ) {
        if width == 0 || height == 0 {
            return;
        }
        self.size = winit::dpi::PhysicalSize::new(width, height);
        self.config.width = width;
        self.config.height = height;
        surface.configure(device, &self.config);
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &wgpu::Surface,
    ) -> Result<(), wgpu::SurfaceError> {
        let frame = surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine-frame-encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine-main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            let pipeline = self.pipeline.as_ref().expect("pipeline not initialized");
            rpass.set_pipeline(pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buffer.as_ref().expect("vb").slice(..));
            rpass.set_index_buffer(
                self.index_buffer.as_ref().expect("ib").slice(..),
                wgpu::IndexFormat::Uint16,
            );
            rpass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

impl Default for Renderer {
    fn default() -> Self {
        panic!("Renderer must be constructed via Renderer::new");
    }
}
