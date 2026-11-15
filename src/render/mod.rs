//! Renderer: wgpu rendering with sprite batching, a camera, and bitmap text.
//!
//! Games queue rectangles, sprites, and text in draw order each frame.
//! Consecutive quads sharing a texture coalesce into a single indexed draw span,
//! preserving exact painter's-algorithm layering with minimal draw calls and
//! bounded power-of-two GPU buffers.

use image::RgbaImage;
use wgpu::util::DeviceExt;

pub mod font;

/// Clear color used each frame (RGBA, 0..=1).
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.1,
    g: 0.1,
    b: 0.12,
    a: 1.0,
};

/// Texture id reserved for solid (untextured) rectangles.
pub const SOLID_TEXTURE: u32 = 0;
/// Texture id of the built-in font atlas.
pub const FONT_TEXTURE: u32 = 1;

/// Camera: world-space centre plus the visible height in world units.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub x: f32,
    pub y: f32,
    pub view_height: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            view_height: 20.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
    uv: [f32; 2],
}

const SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    out.uv = in.uv;
    return out;
}

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(tex, samp, in.uv);
    return sampled * in.color;
}
"#;

/// Per-texture GPU resources.
struct TextureEntry {
    #[allow(dead_code)]
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

/// Coalesced index span referencing a single texture.
#[derive(Clone, Copy, Debug)]
struct DrawSpan {
    texture: u32,
    start: u32,
    count: u32,
}

/// GPU texture created from an image.
pub struct GpuTexture {
    pub width: u32,
    pub height: u32,
    pub view: wgpu::TextureView,
}

pub struct Renderer {
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    #[allow(dead_code)]
    vsync: bool,
    pipeline: Option<wgpu::RenderPipeline>,
    sampler: Option<wgpu::Sampler>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    textures: Vec<TextureEntry>,
    verts: Vec<Vertex>,
    indices: Vec<u32>,
    draws: Vec<DrawSpan>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    camera: Camera,
    clear_color: wgpu::Color,
}

impl Renderer {
    pub fn new(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-texture-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("engine-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x4,
                        2 => Float32x2,
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let mut renderer = Self {
            config,
            size,
            vsync,
            pipeline: Some(pipeline),
            sampler: Some(sampler),
            bind_group_layout: Some(bind_group_layout),
            textures: Vec::new(),
            verts: Vec::with_capacity(4096),
            indices: Vec::with_capacity(6144),
            draws: Vec::with_capacity(32),
            vertex_buffer: None,
            index_buffer: None,
            camera: Camera::default(),
            clear_color: CLEAR_COLOR,
        };

        // Texture 0: solid white pixel.
        let mut white = RgbaImage::new(1, 1);
        white.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
        renderer.create_texture(device, queue, &white);

        // Texture 1: font atlas.
        let atlas = font::build_atlas();
        renderer.create_texture(device, queue, &atlas);

        renderer
    }

    /// Upload an image as a new texture and return its id.
    pub fn create_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &RgbaImage,
    ) -> u32 {
        let width = img.width().max(1);
        let height = img.height().max(1);
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("engine-texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(img.as_raw()),
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-texture-bind-group"),
            layout: self.bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(self.sampler.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        self.textures.push(TextureEntry { view, bind_group });
        (self.textures.len() - 1) as u32
    }

    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

    pub fn camera(&self) -> Camera {
        self.camera
    }

    pub fn set_clear_color(&mut self, color: wgpu::Color) {
        self.clear_color = color;
    }

    /// Surface size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.size.width, self.size.height)
    }

    /// Convert a world-space point to screen pixels (origin top-left, +Y down).
    pub fn world_to_screen(&self, wx: f32, wy: f32) -> (f32, f32) {
        world_to_screen_with(self.camera, self.size.width, self.size.height, wx, wy)
    }

    /// Convert screen pixels (origin top-left, +Y down) to world space.
    pub fn screen_to_world(&self, px: f32, py: f32) -> (f32, f32) {
        screen_to_world_with(self.camera, self.size.width, self.size.height, px, py)
    }

    fn world_to_clip(&self, wx: f32, wy: f32) -> [f32; 2] {
        world_to_clip_with(self.camera, self.size.width, self.size.height, wx, wy)
    }

    /// Queue a solid rectangle centred at `(x, y)` in world space.
    pub fn add_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        self.add_quad(SOLID_TEXTURE, [0.0, 0.0, 1.0, 1.0], x, y, w, h, color);
    }

    /// Queue a textured sprite centred at `(x, y)` in world space.
    ///
    /// `uv` is a normalized atlas rectangle `[u0, v0, u1, v1]`.
    pub fn add_sprite(
        &mut self,
        texture: u32,
        uv: [f32; 4],
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: [f32; 4],
    ) {
        self.add_quad(texture, uv, x, y, w, h, color);
    }

    fn push_quad_clip(
        &mut self,
        texture: u32,
        p0: [f32; 2],
        p1: [f32; 2],
        p2: [f32; 2],
        p3: [f32; 2],
        uv: [f32; 4],
        color: [f32; 4],
    ) {
        if (texture as usize) >= self.textures.len() {
            return;
        }
        let base = self.verts.len() as u32;
        let idx_start = self.indices.len() as u32;
        self.verts.extend_from_slice(&[
            Vertex {
                position: p0,
                color,
                uv: [uv[0], uv[1]],
            },
            Vertex {
                position: p1,
                color,
                uv: [uv[2], uv[1]],
            },
            Vertex {
                position: p2,
                color,
                uv: [uv[0], uv[3]],
            },
            Vertex {
                position: p3,
                color,
                uv: [uv[2], uv[3]],
            },
        ]);
        // Counter-clockwise winding in clip space:
        // p0 (top-left), p2 (bottom-left), p1 (top-right), p3 (bottom-right)
        self.indices
            .extend_from_slice(&[base, base + 2, base + 1, base + 2, base + 3, base + 1]);

        if let Some(last) = self.draws.last_mut()
            && last.texture == texture
        {
            last.count += 6;
        } else {
            self.draws.push(DrawSpan {
                texture,
                start: idx_start,
                count: 6,
            });
        }
    }

    fn add_quad(
        &mut self,
        texture: u32,
        uv: [f32; 4],
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: [f32; 4],
    ) {
        let hw = w * 0.5;
        let hh = h * 0.5;
        let p0 = self.world_to_clip(x - hw, y - hh);
        let p1 = self.world_to_clip(x + hw, y - hh);
        let p2 = self.world_to_clip(x - hw, y + hh);
        let p3 = self.world_to_clip(x + hw, y + hh);
        self.push_quad_clip(texture, p0, p1, p2, p3, uv, color);
    }

    /// Queue text in screen space (pixels, origin top-left).
    pub fn draw_text(&mut self, text: &str, x: f32, y: f32, scale: f32, color: [f32; 4]) {
        let cw = font::GLYPH_WIDTH as f32 * scale;
        let ch = font::GLYPH_HEIGHT as f32 * scale;
        let step = font::GLYPH_WIDTH as f32 * scale + scale;
        let mut cursor = x;
        for ch_char in text.chars() {
            let uv = font::char_uv(ch_char);
            self.add_screen_quad(FONT_TEXTURE, uv, cursor, y, cw, ch, color);
            cursor += step;
        }
    }

    /// Width in pixels of `text` rendered at `scale`.
    pub fn text_width(&self, text: &str, scale: f32) -> f32 {
        font::text_width(text, scale)
    }

    /// Height in pixels of a text line rendered at `scale`.
    pub fn text_height(&self, scale: f32) -> f32 {
        font::text_height(scale)
    }

    /// Queue a screen-space textured sprite (pixels, origin top-left).
    pub fn add_sprite_screen(
        &mut self,
        texture: u32,
        uv: [f32; 4],
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: [f32; 4],
    ) {
        self.add_screen_quad(texture, uv, x, y, w, h, color);
    }

    fn add_screen_quad(
        &mut self,
        texture: u32,
        uv: [f32; 4],
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: [f32; 4],
    ) {
        let (sw, sh) = (
            self.size.width.max(1) as f32,
            self.size.height.max(1) as f32,
        );
        let to_clip = |px: f32, py: f32| [px / sw * 2.0 - 1.0, 1.0 - py / sh * 2.0];
        let p0 = to_clip(x, y);
        let p1 = to_clip(x + w, y);
        let p2 = to_clip(x, y + h);
        let p3 = to_clip(x + w, y + h);
        self.push_quad_clip(texture, p0, p1, p2, p3, uv, color);
    }

    /// Queue a screen-space rectangle (pixels, origin top-left).
    pub fn add_rect_screen(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        self.add_screen_quad(SOLID_TEXTURE, [0.0, 0.0, 1.0, 1.0], x, y, w, h, color);
    }

    fn ensure_buffers(&mut self, device: &wgpu::Device) {
        if self.verts.is_empty() || self.indices.is_empty() {
            return;
        }
        let needed_vb = (self.verts.len() * std::mem::size_of::<Vertex>()) as u64;
        if self
            .vertex_buffer
            .as_ref()
            .map(|b| b.size() < needed_vb)
            .unwrap_or(true)
        {
            let cap = needed_vb.next_power_of_two().max(32 * 1024);
            self.vertex_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine-batch-vb"),
                size: cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let needed_ib = (self.indices.len() * std::mem::size_of::<u32>()) as u64;
        if self
            .index_buffer
            .as_ref()
            .map(|b| b.size() < needed_ib)
            .unwrap_or(true)
        {
            let cap = needed_ib.next_power_of_two().max(16 * 1024);
            self.index_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine-batch-ib"),
                size: cap,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
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
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.size = winit::dpi::PhysicalSize::new(self.config.width, self.config.height);
        surface.configure(device, &self.config);
        log::info!(
            "surface resized to {}x{}",
            self.config.width,
            self.config.height
        );
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

        self.ensure_buffers(device);
        if !self.verts.is_empty() && !self.indices.is_empty() {
            if let Some(vb) = self.vertex_buffer.as_ref() {
                queue.write_buffer(vb, 0, bytemuck::cast_slice(&self.verts));
            }
            if let Some(ib) = self.index_buffer.as_ref() {
                queue.write_buffer(ib, 0, bytemuck::cast_slice(&self.indices));
            }
        }

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
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if !self.draws.is_empty()
                && let (Some(vb), Some(ib)) =
                    (self.vertex_buffer.as_ref(), self.index_buffer.as_ref())
            {
                let pipeline = self.pipeline.as_ref().expect("pipeline not initialized");
                rpass.set_pipeline(pipeline);
                rpass.set_vertex_buffer(0, vb.slice(..));
                rpass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);

                for span in &self.draws {
                    rpass.set_bind_group(0, &self.textures[span.texture as usize].bind_group, &[]);
                    rpass.draw_indexed(span.start..(span.start + span.count), 0, 0..1);
                }
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        self.verts.clear();
        self.indices.clear();
        self.draws.clear();

        Ok(())
    }
}

impl Default for Renderer {
    fn default() -> Self {
        panic!("Renderer must be constructed via Renderer::new");
    }
}

fn world_to_clip_with(cam: Camera, width: u32, height: u32, wx: f32, wy: f32) -> [f32; 2] {
    let sw = width.max(1) as f32;
    let sh = height.max(1) as f32;
    let aspect = sw / sh;
    let view_w = cam.view_height * aspect;
    [
        (wx - cam.x) / (view_w * 0.5),
        -(wy - cam.y) / (cam.view_height * 0.5),
    ]
}

fn world_to_screen_with(cam: Camera, width: u32, height: u32, wx: f32, wy: f32) -> (f32, f32) {
    let sw = width.max(1) as f32;
    let sh = height.max(1) as f32;
    let aspect = sw / sh;
    let view_w = cam.view_height * aspect;
    let px = (wx - cam.x) / view_w + 0.5;
    let py = (wy - cam.y) / cam.view_height + 0.5;
    (px * sw, py * sh)
}

fn screen_to_world_with(cam: Camera, width: u32, height: u32, px: f32, py: f32) -> (f32, f32) {
    let sw = width.max(1) as f32;
    let sh = height.max(1) as f32;
    let aspect = sw / sh;
    let view_w = cam.view_height * aspect;
    let fx = px / sw - 0.5;
    let fy = py / sh - 0.5;
    (cam.x + fx * view_w, cam.y + fy * cam.view_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_world_round_trip_matches_clip_space() {
        let cam = Camera {
            x: 10.0,
            y: -5.0,
            view_height: 20.0,
        };
        let (wx, wy) = screen_to_world_with(cam, 800, 600, 200.0, 150.0);
        let (px, py) = world_to_screen_with(cam, 800, 600, wx, wy);
        assert!((px - 200.0).abs() < 1e-3);
        assert!((py - 150.0).abs() < 1e-3);

        // Top of screen (py=0) maps to clip y=+1.0; bottom (py=600) maps to clip y=-1.0.
        let (_, top_wy) = screen_to_world_with(cam, 800, 600, 400.0, 0.0);
        let clip_top = world_to_clip_with(cam, 800, 600, cam.x, top_wy);
        assert!((clip_top[1] - 1.0).abs() < 1e-4);

        let (_, bot_wy) = screen_to_world_with(cam, 800, 600, 400.0, 600.0);
        let clip_bot = world_to_clip_with(cam, 800, 600, cam.x, bot_wy);
        assert!((clip_bot[1] + 1.0).abs() < 1e-4);
    }
}
