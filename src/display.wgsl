// Draws the simulation state to the window: u through the Turbo colour map,
// letterboxed so the grid keeps its aspect ratio.

struct View {
  window: vec2<f32>, // surface size in pixels
  grid: vec2<f32>,   // NX, NY
}

@group(0) @binding(0) var state_tex : texture_2d<f32>;
@group(0) @binding(1) var<uniform> view : View;

// One triangle that covers the whole screen; no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
  let x = f32(i32(i & 1u) * 4 - 1); // -1, 3, -1
  let y = f32(i32(i >> 1u) * 4 - 1); // -1, -1, 3
  return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag : vec4<f32>) -> @location(0) vec4<f32> {
  // letterbox: largest scale that fits the grid in the window
  let scale = min(view.window.x / view.grid.x, view.window.y / view.grid.y);
  let offset = 0.5 * (view.window - view.grid * scale);
  let g = (frag.xy - offset) / scale; // pixel -> grid coordinates

  if (g.x < 0.0 || g.y < 0.0 || g.x >= view.grid.x || g.y >= view.grid.y) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // outside the grid
  }

  // nearest texel; textureLoad needs no sampler and works on rgba32float
  let u = textureLoad(state_tex, vec2<i32>(g), 0).r;
  return vec4<f32>(turbo(u), 1.0);
}

// Polynomial approximation of Google's Turbo colour map, input in [0, 1].
fn turbo(x_in : f32) -> vec3<f32> {
  let x = clamp(x_in, 0.0, 1.0);
  let r = vec4<f32>(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  let g = vec4<f32>(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  let b = vec4<f32>(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  let r2 = vec2<f32>(-152.94239396, 59.28637943);
  let g2 = vec2<f32>(4.27729857, 2.82956604);
  let b2 = vec2<f32>(-89.90310912, 27.34824973);
  let v4 = vec4<f32>(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  return vec3<f32>(
    dot(v4, r) + dot(v2, r2),
    dot(v4, g) + dot(v2, g2),
    dot(v4, b) + dot(v2, b2),
  );
}
