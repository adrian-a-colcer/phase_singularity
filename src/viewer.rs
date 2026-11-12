//! Live viewer: a winit window that steps the simulation and draws u every frame.

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::gpu::Gpu;
use crate::rd::{DT, FHNParameters, NX, NY, Simulation};

pub fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App { state: None };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    state: Option<State>,
}

struct State {
    gpu: Gpu,
    sim: Simulation,
    renderer: Renderer,
    steps_per_frame: u32,
    paused: bool,
    // for the FPS readout in the title bar
    last_title_update: Instant,
    frames_since_update: u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return; // desktop calls this once; mobile/web can call it again
        }

        // 1. window, wrapped in Arc so the surface and Gpu can share it
        let attrs = Window::default_attributes()
            .with_title("FitzHugh–Nagumo")
            .with_inner_size(winit::dpi::LogicalSize::new(768.0, 768.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        // 2. GPU from the window (async -> block_on, winit callbacks aren't async)
        let gpu = pollster::block_on(Gpu::new(window)).expect("GPU init");

        // 3. simulation and renderer from the *window's* device
        let sim = Simulation::new(&gpu.device, &gpu.queue, FHNParameters::default());
        let renderer = Renderer::new(&gpu, &sim);

        self.state = Some(State {
            gpu,
            sim,
            renderer,
            steps_per_frame: 64,
            paused: false,
            last_title_update: Instant::now(),
            frames_since_update: 0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                state.gpu.resize(size.width, size.height);
                state.renderer.update_view(&state.gpu);
            }

            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key: PhysicalKey::Code(key), state: ElementState::Pressed, .. },
                ..
            } => match key {
                KeyCode::Escape => event_loop.exit(),
                KeyCode::Space => state.paused = !state.paused,
                KeyCode::ArrowUp => state.steps_per_frame = (state.steps_per_frame * 2).min(4096),
                KeyCode::ArrowDown => state.steps_per_frame = (state.steps_per_frame / 2).max(1),
                KeyCode::KeyR => state.sim.reset(&state.gpu.queue),
                _ => {}
            },

            WindowEvent::RedrawRequested => {
                if let Err(e) = state.frame() {
                    log::error!("{e}");
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }

    // keep frames coming
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.gpu.window.request_redraw();
        }
    }
}

impl State {
    fn frame(&mut self) -> anyhow::Result<()> {
        // 1. next image to draw into
        let frame = match self.gpu.surface.get_current_texture() {
            Ok(frame) => frame,
            // resize, display change, sleep: reconfigure and try next frame
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let (w, h) = (self.gpu.config.width, self.gpu.config.height);
                self.gpu.resize(w, h);
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(()), // skip this frame
            Err(e) => anyhow::bail!("surface error: {e}"),
        };
        let target = frame.texture.create_view(&Default::default());

        // 2. one encoder: N simulation steps, then the draw
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        if !self.paused {
            self.sim.step(&mut encoder, self.steps_per_frame);
        }
        self.renderer.draw(&mut encoder, &target, self.sim.current());

        // 3. submit and show
        self.gpu.queue.submit([encoder.finish()]);
        self.gpu.window.pre_present_notify();
        frame.present();

        self.update_title();
        Ok(())
    }

    fn update_title(&mut self) {
        self.frames_since_update += 1;
        let elapsed = self.last_title_update.elapsed().as_secs_f32();
        if elapsed >= 0.5 {
            let fps = self.frames_since_update as f32 / elapsed;
            let t = self.sim.steps_taken() as f32 * DT;
            let status = if self.paused { "  [paused]" } else { "" };
            self.gpu.window.set_title(&format!(
                "FitzHugh–Nagumo  t = {t:.0}  |  {} steps/frame  |  {fps:.0} fps{status}",
                self.steps_per_frame
            ));
            self.last_title_update = Instant::now();
            self.frames_since_update = 0;
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ViewUniform {
    window: [f32; 2],
    grid: [f32; 2],
}

struct Renderer {
    pipeline: wgpu::RenderPipeline,
    view_buffer: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2], // one per simulation texture
}

impl Renderer {
    fn new(gpu: &Gpu, sim: &Simulation) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("display.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Display"),
            layout: None, // derived from the shader
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // vertices come from vertex_index
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let view_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("View Uniform"),
            contents: bytemuck::bytes_of(&view_uniform(gpu)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = pipeline.get_bind_group_layout(0);
        let make_bind_group = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Display bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1, resource: view_buffer.as_entire_binding() },
                ],
            })
        };
        let views = sim.views();
        let bind_groups = [make_bind_group(&views[0]), make_bind_group(&views[1])];

        Self { pipeline, view_buffer, bind_groups }
    }

    /// Call after a resize so the letterboxing uses the new window size.
    fn update_view(&self, gpu: &Gpu) {
        gpu.queue.write_buffer(&self.view_buffer, 0, bytemuck::bytes_of(&view_uniform(gpu)));
    }

    fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, current: usize) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Display pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[current], &[]);
        pass.draw(0..3, 0..1); // the full-screen triangle
    }
}

fn view_uniform(gpu: &Gpu) -> ViewUniform {
    ViewUniform {
        window: [gpu.config.width as f32, gpu.config.height as f32],
        grid: [NX as f32, NY as f32],
    }
}