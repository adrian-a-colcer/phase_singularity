use flume::bounded;
use wgpu::{BufferUsages, util::{BufferInitDescriptor, DeviceExt}, wgc::validation::BindingTypeName::Buffer};
use bytemuck::{Pod, Zeroable, bytes_of};

// taken from reaction_diffusion_wgpu 
// https://github.com/ArminGEtemad/reaction_diffusion_wgpu.git
// time 
// lives in group 0 binding 0 (FHN shader)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TimeUniform {
    // 16 byte alignment needed
    dt: f32, // 4 byte
    _pad: [f32; 3], // 12 byte
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FHNParameters {
    pub du_rate: f32,
    pub a: f32,
    pub b: f32,
    pub c1: f32,
    pub c2: f32,
    pub d: f32,
    pub _pad: [f32; 2],
}

pub async fn sim() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(&Default::default());
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default.default()).await.unwrap();

    let shader = device.create_shader_module(wgpu::include_wgsl!("FitzHughNagumo.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipeLineDescriptor {
        label: Some("Reaction-Diffusion Simulator"),
        layout: None,
        module: &shader,
        entry_point: Some("main_euler"),
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let time_uniform = TimeUniform {
        dt: 0.0,
        _pad: [0.0; 3],
    };

    let time_buffer = device.create_buffer(&wgpu::BufferInitDescriptor {
        label: Some("Time Uniform Buffer"),
        contents: bytemuck::bytes_of(&time_uniform),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
    
    let params_buffer =
}
