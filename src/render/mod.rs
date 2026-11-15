//! Renderer: minimal wgpu-based rendering with dynamic colored-rect batching.
//!
//! Games queue rectangles each frame via [`Renderer::add_rect`]; the renderer
//! batches them into a single draw call. Fixed memory footprint.

use wgpu::util::DeviceExt;

/// Clear color used each frame (RGBA, 0..=1).
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.05,
    g: 0.05,
    b: 0.08,
    a: 1.0,
};

/// Maximum number of rectangles that can be queued per frame.
const MAX_RECTS: usize = 4096;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

/// A single rectangle queued for drawing, in clip space (-1..=1).
#[derive(Clone, Copy)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub color: [f32; 4],
}

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
    rects: Vec<Rect>,
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

        // Dynamic vertex buffer: updated every frame with queued rects.
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine-rect-vertex-buffer"),
            size: (MAX_RECTS * 4 * std::mem::size_of::<Vertex>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Static index buffer: same quad index pattern repeated for every rect slot.
        let mut indices = Vec::with_capacity(MAX_RECTS * 6);
        for i in 0..MAX_RECTS {
            let base = (i * 4) as u16;
            indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
        }
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine-rect-index-buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            config,
            size,
            vsync,
            pipeline: Some(pipeline),
            vertex_buffer: Some(vertex_buffer),
            index_buffer: Some(index_buffer),
            rects: Vec::with_capacity(MAX_RECTS),
        }
    }

    /// Queue a rectangle for drawing this frame. Coordinates are in clip space
    /// where (-1, -1) is bottom-left and (1, 1) is top-right. `(x, y)` is the
    /// center of the rectangle.
    pub fn add_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        if self.rects.len() < MAX_RECTS {
            self.rects.push(Rect { x, y, w, h, color });
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
        // Build vertices from queued rects.
        let rect_count = self.rects.len();
        let mut vertices = Vec::with_capacity(rect_count * 4);
        for rect in &self.rects {
            let x0 = rect.x - rect.w * 0.5;
            let x1 = rect.x + rect.w * 0.5;
            let y0 = rect.y - rect.h * 0.5;
            let y1 = rect.y + rect.h * 0.5;
            let c = rect.color;
            vertices.push(Vertex { position: [x0, y0], color: c });
            vertices.push(Vertex { position: [x1, y0], color: c });
            vertices.push(Vertex { position: [x0, y1], color: c });
            vertices.push(Vertex { position: [x1, y1], color: c });
        }

        let frame = surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine-frame-encoder"),
        });

        // Upload vertex data.
        if !vertices.is_empty() {
            queue.write_buffer(
                self.vertex_buffer.as_ref().expect("vertex buffer"),
                0,
                bytemuck::cast_slice(&vertices),
            );
        }

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
            rpass.set_vertex_buffer(
                0,
                self.vertex_buffer.as_ref().expect("vb").slice(..),
            );
            rpass.set_index_buffer(
                self.index_buffer.as_ref().expect("ib").slice(..),
                wgpu::IndexFormat::Uint16,
            );
            let index_count = (rect_count * 6) as u32;
            if index_count > 0 {
                rpass.draw_indexed(0..index_count, 0, 0..1);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        // Reset for next frame.
        self.rects.clear();

        Ok(())
    }
}

impl Default for Renderer {
    fn default() -> Self {
        panic!("Renderer must be constructed via Renderer::new");
    }
}
