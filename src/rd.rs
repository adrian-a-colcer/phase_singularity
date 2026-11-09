use flume::bounded;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub async fn sim() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(&Default::default());
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default.default()).await.unwrap();

    let shader = device.create_shader_module(wgpu::include_wgsl!("FitzHughNagumo.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipeLineDescriptor {
        label: Some("Reaction-Diffusion Simulator"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let in
}
