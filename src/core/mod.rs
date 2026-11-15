//! Core engine: application lifecycle, window, input, and the main run loop.

use std::time::Instant;

use log::info;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::Key;
use winit::window::Window;

use crate::asset::AssetLoader;
use crate::audio::Audio;
use crate::memory::MemoryBudget;
use crate::render::Renderer;

/// Snapshot of input for the current frame.
#[derive(Default, Clone)]
pub struct InputState {
    pub keys_just_pressed: Vec<Key>,
    /// Keys currently held down (updated each frame from press/release events).
    pub keys_held: Vec<Key>,
    /// Mouse position in physical pixels, origin at the top-left of the window.
    pub mouse_position: (f32, f32),
    /// Mouse buttons currently held down.
    pub mouse_held: Vec<winit::event::MouseButton>,
    /// Mouse buttons pressed during this frame.
    pub mouse_just_pressed: Vec<winit::event::MouseButton>,
}

/// Context handed to the game each frame.
pub struct RunContext<'a, 'b> {
    pub delta: f32,
    pub rss_mb: f32,
    pub input: &'a InputState,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub renderer: &'b mut Renderer,
}

/// Context handed to the game at initialization.
pub struct InitContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub renderer: &'a mut Renderer,
}

/// Trait implemented by the game. The engine owns the loop; the game owns logic.
pub trait App {
    fn init(&mut self, _ctx: &mut InitContext) {}
    fn update(&mut self, _ctx: &mut RunContext) {}
    fn render(&mut self, _ctx: &mut RunContext) {}
}

/// Builder for constructing an [Engine].
pub struct AppBuilder {
    title: String,
    width: u32,
    height: u32,
    vsync: bool,
    memory_budget_mb: usize,
}

impl AppBuilder {
    pub fn new() -> Self {
        Self {
            title: "Custom Game".to_string(),
            width: 800,
            height: 600,
            vsync: true,
            memory_budget_mb: 100,
        }
    }

    pub fn title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn vsync(mut self, enabled: bool) -> Self {
        self.vsync = enabled;
        self
    }

    pub fn memory_budget_mb(mut self, mb: usize) -> Self {
        self.memory_budget_mb = mb;
        self
    }

    pub fn build<A: App + 'static>(self, app: A) -> Engine<A> {
        Engine::new(self, app)
    }
}

impl Default for AppBuilder {
    fn default() -> Self {
        Self::new()
    }
}

struct BuilderConfig {
    title: String,
    width: u32,
    height: u32,
    vsync: bool,
    #[allow(dead_code)]
    memory_budget_mb: usize,
}

/// The engine. Owns the window, GPU device, and all subsystems.
pub struct Engine<A: App> {
    app: A,
    config: BuilderConfig,
    /// Subsystems (lazy-initialized after the window resumes).
    adapter: Option<wgpu::Adapter>,
    window: Option<Window>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    renderer: Option<Renderer>,
    #[allow(dead_code)]
    assets: AssetLoader,
    #[allow(dead_code)]
    audio: Audio,
    memory: MemoryBudget,
    input: InputState,
    last_frame: Option<Instant>,
}

impl<A: App + 'static> Engine<A> {
    fn new(config: AppBuilder, app: A) -> Self {
        Self {
            app,
            config: BuilderConfig {
                title: config.title,
                width: config.width,
                height: config.height,
                vsync: config.vsync,
                memory_budget_mb: config.memory_budget_mb,
            },
            window: None,
            surface: None,
            device: None,
            queue: None,
            adapter: None,
            renderer: None,
            assets: AssetLoader::new(),
            audio: Audio::new(),
            memory: MemoryBudget::new(config.memory_budget_mb),
            input: InputState::default(),
            last_frame: None,
        }
    }

    /// Build and run the engine. Blocks until the window closes.
    pub fn run(self) {
        let event_loop = winit::event_loop::EventLoop::new().expect("failed to create event loop");
        let mut engine = self;
        event_loop.run_app(&mut engine).expect("run loop failed");
    }
}

impl<A: App + 'static> ApplicationHandler for Engine<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = winit::window::Window::default_attributes()
            .with_title(&self.config.title)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ));
        let window = event_loop.create_window(attrs).expect("failed to create window");

        info!(
            "engine window created {}x{}",
            self.config.width, self.config.height
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(&window)
            .expect("failed to create surface");
        // Widen the surface lifetime to 'static. This is sound because the surface holds
        // the OS window handle (independent of the Rust `Window` value's location) and the
        // Engine struct drops `surface` before `window` (declared earlier), so the surface
        // never outlives the window it references.
        let surface =
            unsafe { std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface) };
        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .expect("no GPU adapter found");
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::downlevel_defaults(),
                        label: Some("engine-device"),
                        memory_hints: wgpu::MemoryHints::default(),
                    },
                    None,
                )
                .await
                .expect("failed to create GPU device");
            (adapter, device, queue)
        });

        let mut renderer = Renderer::new(
            &adapter,
            &device,
            &queue,
            &surface,
            self.config.width,
            self.config.height,
            self.config.vsync,
        );
        let mut init_ctx = InitContext {
            device: &device,
            queue: &queue,
            renderer: &mut renderer,
        };
        self.app.init(&mut init_ctx);

        self.window = Some(window);
        self.surface = Some(surface);
        self.adapter = Some(adapter);
        self.device = Some(device);
        self.queue = Some(queue);
        self.renderer = Some(renderer);
        self.last_frame = Some(Instant::now());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let key = event.logical_key.clone();
                let is_pressed = event.state.is_pressed();
                let was_held = self.input.keys_held.contains(&key);
                if is_pressed {
                    if !was_held {
                        self.input.keys_just_pressed.push(key.clone());
                    }
                    if !was_held {
                        self.input.keys_held.push(key);
                    }
                } else {
                    self.input.keys_held.retain(|k| k != &key);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.mouse_position = (position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let is_pressed = state.is_pressed();
                let was_held = self.input.mouse_held.contains(&button);
                if is_pressed {
                    if !was_held {
                        self.input.mouse_just_pressed.push(button);
                    }
                    if !was_held {
                        self.input.mouse_held.push(button);
                    }
                } else {
                    self.input.mouse_held.retain(|b| b != &button);
                }
            }
            WindowEvent::Resized(size) => {
                if let (Some(surface), Some(device), Some(queue), Some(renderer)) = (
                    &mut self.surface,
                    &mut self.device,
                    &mut self.queue,
                    &mut self.renderer,
                ) {
                    renderer.resize(device, queue, surface, size.width, size.height);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window = match self.window.take() {
            Some(w) => w,
            None => return,
        };
        let surface = match self.surface.take() {
            Some(s) => s,
            None => {
                self.window = Some(window);
                return;
            }
        };
        let device = match self.device.take() {
            Some(d) => d,
            None => {
                self.window = Some(window);
                self.surface = Some(surface);
                return;
            }
        };
        let queue = match self.queue.take() {
            Some(q) => q,
            None => {
                self.window = Some(window);
                self.surface = Some(surface);
                self.device = Some(device);
                return;
            }
        };
        let mut renderer = match self.renderer.take() {
            Some(r) => r,
            None => {
                self.window = Some(window);
                self.surface = Some(surface);
                self.device = Some(device);
                self.queue = Some(queue);
                return;
            }
        };

        let now = Instant::now();
        let delta = self
            .last_frame
            .map(|f| now.duration_since(f).as_secs_f32())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        self.last_frame = Some(now);

        let rss_mb = self.memory.tick();

        let mut ctx = RunContext {
            delta,
            rss_mb,
            input: &self.input,
            device: &device,
            queue: &queue,
            renderer: &mut renderer,
        };

        self.app.update(&mut ctx);
        self.app.render(&mut ctx);

        // Reset per-frame input after update and render have consumed it.
        self.input.keys_just_pressed.clear();
        self.input.mouse_just_pressed.clear();

        // Present.
        match renderer.render(&device, &queue, &surface) {
            Ok(()) => {}
            Err(wgpu::SurfaceError::Lost) => {
                renderer.resize(&device, &queue, &surface, self.config.width, self.config.height);
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                log::warn!("surface out of memory - engine exceeded GPU budget");
            }
            Err(e) => log::warn!("surface error: {:?}", e),
        }

        window.request_redraw();

        // Put subsystems back.
        self.window = Some(window);
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.renderer = Some(renderer);
    }
}
