use flume::bounded;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use bytemuck::{Pod, Zeroable}; 

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TimeUniform {
    pub dt: f32,
    pub _pad: [f32; 3],
}

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

// ---------------------------------------------------------------------------
// Simulation Function
// ---------------------------------------------------------------------------

pub async fn sim() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(&Default::default());
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    let shader = device.create_shader_module(wgpu::include_wgsl!("FitzHughNagumo.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Reaction-Diffusion Simulator"),
        layout: None,
        module: &shader,
        entry_point: Some("main_euler"), // points to @compute fn main_euler
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    // Create the uniform buffers using DeviceExt
    let time_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Time Uniform Buffer"),
        contents: bytemuck::cast_slice(&[TimeUniform {
            dt: 0.05,
            _pad: [0.0; 3],
        }]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("FHN Parameters Buffer"),
        contents: bytemuck::cast_slice(&[FHNParameters {
            du_rate: 1.0,
            a: 0.1,
            b: 0.01,
            c1: 1.0,
            c2: 1.0,
            d: 1.0,
            _pad: [0.0; 2],
        }]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    Ok(())
}
