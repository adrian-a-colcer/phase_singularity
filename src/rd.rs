use bytemuck::{Pod, Zeroable};
use flume::bounded;
use wgpu::util::{BufferInitDescriptor, DeviceExt, TextureDataOrder};

// ---------------------------------------------------------------------------
// Uniforms (must match the structs in FitzHughNagumo.wgsl)
// ---------------------------------------------------------------------------

// group 0 binding 0
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TimeUniform {
    pub dt: f32,
    pub _pad: [f32; 3], // uniform buffers: keep sizes a multiple of 16 bytes
}

// group 0 binding 1
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FHNParameters {
    pub du_rate: f32,
    pub a: f32,
    pub b: f32,
    pub c1: f32,
    pub c2: f32,
    pub d: f32,
    pub _pad: [f32; 2],
}

impl Default for FHNParameters {
    fn default() -> Self {
        Self { du_rate: 1.0, a: 0.1, b: 0.01, c1: 1.0, c2: 1.0, d: 1.0, _pad: [0.0; 2] }
    }
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

pub const NX: u32 = 512;
pub const NY: u32 = 512;
pub const DT: f32 = 0.05;
const WORKGROUP: u32 = 16; // must match @workgroup_size(16, 16) in the shader
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const BYTES_PER_TEXEL: u32 = 16; // 4 x f32

/// One texel of state: (u, v, unused, unused).
pub type Texel = [f32; 4];

// ---------------------------------------------------------------------------
// Simulation: owns the GPU resources, but not the device.
// Works the same headless or inside the live viewer.
// ---------------------------------------------------------------------------

pub struct Simulation {
    pipeline: wgpu::ComputePipeline,
    textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    bind_groups: [wgpu::BindGroup; 2], // [0] reads A writes B, [1] reads B writes A
    _time_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,
    current: usize, // index of the texture holding the newest state
    steps_taken: u64,
}

impl Simulation {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, params: FHNParameters) -> Self {
        // --- pipeline (layout derived from the shader) ---
        let shader = device.create_shader_module(wgpu::include_wgsl!("FitzHughNagumo.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reaction-Diffusion Simulator"),
            layout: None,
            module: &shader,
            entry_point: Some("main_euler"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        // --- uniform buffers ---
        let time_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Time Uniform Buffer"),
            contents: bytemuck::cast_slice(&[TimeUniform { dt: DT, _pad: [0.0; 3] }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("FHN Parameters Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- two textures for ping-pong ---
        let initial = initial_state(NX, NY);
        let make_texture = |label: &str| {
            device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: NX, height: NY, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING // read as tex_in / by the display
                        | wgpu::TextureUsages::STORAGE_BINDING   // written as tex_out
                        | wgpu::TextureUsages::COPY_SRC          // read back to the CPU
                        | wgpu::TextureUsages::COPY_DST,         // reset
                    view_formats: &[],
                },
                TextureDataOrder::LayerMajor,
                bytemuck::cast_slice(&initial),
            )
        };
        let textures = [make_texture("State A"), make_texture("State B")];
        let views = [
            textures[0].create_view(&Default::default()),
            textures[1].create_view(&Default::default()),
        ];

        // --- two bind groups ---
        let layout = pipeline.get_bind_group_layout(0);
        let make_bind_group = |src: &wgpu::TextureView, dst: &wgpu::TextureView, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: time_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(dst) },
                ],
            })
        };
        let bind_groups = [
            make_bind_group(&views[0], &views[1], "A -> B"),
            make_bind_group(&views[1], &views[0], "B -> A"),
        ];

        Self {
            pipeline,
            textures,
            views,
            bind_groups,
            _time_buffer: time_buffer,
            _params_buffer: params_buffer,
            current: 0,
            steps_taken: 0,
        }
    }

    /// Record `n` steps into the caller's encoder. Nothing runs until the
    /// caller submits, so steps and drawing can share one submission.
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder, n: u32) {
        for _ in 0..n {
            // pass boundaries guarantee step k finished writing before step k+1 reads
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FHN step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.current], &[]);
            pass.dispatch_workgroups(NX.div_ceil(WORKGROUP), NY.div_ceil(WORKGROUP), 1);
            drop(pass);
            self.current = 1 - self.current; // the texture just written is now the newest
        }
        self.steps_taken += n as u64;
    }

    /// Index (0 or 1) of the texture holding the newest state.
    pub fn current(&self) -> usize {
        self.current
    }

    pub fn views(&self) -> &[wgpu::TextureView; 2] {
        &self.views
    }

    pub fn steps_taken(&self) -> u64 {
        self.steps_taken
    }

    /// Put the initial condition back into both textures.
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        let initial = initial_state(NX, NY);
        for tex in &self.textures {
            queue.write_texture(
                tex.as_image_copy(),
                bytemuck::cast_slice(&initial),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(NX * BYTES_PER_TEXEL),
                    rows_per_image: Some(NY),
                },
                tex.size(),
            );
        }
        self.current = 0;
        self.steps_taken = 0;
    }

    /// Copy the newest state to the CPU (blocks until the GPU is done).
    pub fn read_back(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<Vec<Texel>> {
        let latest = &self.textures[self.current];

        // texture-to-buffer copies need each row padded to 256 bytes
        let unpadded_row = NX * BYTES_PER_TEXEL;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(align) * align;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: (padded_row * NY) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            latest.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(NY),
                },
            },
            latest.size(),
        );
        queue.submit([encoder.finish()]);

        let (tx, rx) = bounded(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::Wait)?;
        rx.recv()??;

        let mut state = Vec::with_capacity((NX * NY) as usize);
        {
            let data = staging.slice(..).get_mapped_range();
            for row in data.chunks(padded_row as usize) {
                state.extend_from_slice(bytemuck::cast_slice::<u8, Texel>(&row[..unpadded_row as usize]));
            }
        }
        staging.unmap();
        Ok(state)
    }
}

/// Headless run: own device, no window. Used for PNGs and benchmarks.
pub async fn run_headless(steps: u32) -> anyhow::Result<Vec<Texel>> {
    let instance = wgpu::Instance::new(&Default::default());
    let adapter = instance.request_adapter(&Default::default()).await?;
    let (device, queue) = adapter.request_device(&Default::default()).await?;

    let mut sim = Simulation::new(&device, &queue, FHNParameters::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    sim.step(&mut encoder, steps);
    queue.submit([encoder.finish()]);
    sim.read_back(&device, &queue)
}

/// Cross-field stimulus: u excited on the left half, v refractory on the
/// bottom half. The broken wavefront curls into a spiral.
fn initial_state(nx: u32, ny: u32) -> Vec<Texel> {
    let mut state = vec![[0.0, 0.0, 0.0, 1.0]; (nx * ny) as usize];
    for y in 0..ny {
        for x in 0..nx {
            let i = (y * nx + x) as usize;
            if x < nx / 2 {
                state[i][0] = 1.0; // u
            }
            if y < ny / 2 {
                state[i][1] = 0.5; // v
            }
        }
    }
    state
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Save u as a PNG (dark = resting, bright = excited).
pub fn save_png(path: &str, state: &[Texel], nx: u32, ny: u32) -> anyhow::Result<()> {
    let mut img = image::RgbImage::new(nx, ny);
    for y in 0..ny {
        for x in 0..nx {
            let u = state[(y * nx + x) as usize][0];
            img.put_pixel(x, y, image::Rgb(colormap(u)));
        }
    }
    img.save(path)?;
    Ok(())
}

/// Simple dark-blue -> orange -> yellow ramp for u in roughly [0, 1].
fn colormap(u: f32) -> [u8; 3] {
    let t = u.clamp(0.0, 1.0);
    let r = (255.0 * (1.5 * t).min(1.0)) as u8;
    let g = (255.0 * (1.5 * t - 0.4).clamp(0.0, 1.0)) as u8;
    let b = (90.0 * (1.0 - t) + 40.0 * t) as u8;
    [r, g, b]
}