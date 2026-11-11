struct TimeUniform {
  dt: f32,
};

struct FHNParameters {
  du_rate: f32,
  a: f32,
  b: f32,
  c1: f32,
  c2: f32,
  d: f32,
};

@group(0) @binding(0)
var<uniform> u_time : TimeUniform;

@group(0) @binding(1)
var<uniform> sys_params : FHNParameters;

// Textures: source (step n) and destination (step n + 1)
@group(0) @binding(2)
var tex_in : texture_2d<f32>;

@group(0) @binding(3)
var tex_out : texture_storage_2d<rgba32float, write>;

fn fhn_system_rhs(u: f32, v: f32, lap_u: f32) -> vec2<f32> {
  let du = sys_params.du_rate * lap_u 
         + sys_params.c1 * u * (u - sys_params.a) * (1.0 - u)
         - sys_params.c2 * u * v;

  let dv = sys_params.b * (u - sys_params.d * v);

  return vec2<f32>(du, dv);
};

fn nine_point_stencil(
  center: vec2<f32>, up: vec2<f32>,
  down: vec2<f32>, left: vec2<f32>,
  right: vec2<f32>, up_left: vec2<f32>,
  up_right: vec2<f32>, down_left: vec2<f32>,
  down_right: vec2<f32>,
) -> vec2<f32> {
  let cross_term = 4.0 * (up + down + left + right);
  let diag_term = up_left + up_right + down_left + down_right;
  let laplace = (cross_term + diag_term - 20.0 * center) / 6.0;
  return laplace;
}

fn read_u_v(x_y: vec2<i32>) -> vec2<f32> {
  let dims = textureDimensions(tex_in);
  let x = clamp(x_y.x, 0, i32(dims.x) - 1);
  let y = clamp(x_y.y, 0, i32(dims.y) - 1);
  return textureLoad(tex_in, vec2<i32>(x, y), 0).rg;
}

fn laplacian_u(center_u: f32, x_y: vec2<i32>) -> f32 {
  let up         = read_u_v(x_y + vec2<i32>( 0, -1)).x;
  let down       = read_u_v(x_y + vec2<i32>( 0,  1)).x;
  let left       = read_u_v(x_y + vec2<i32>(-1,  0)).x;
  let right      = read_u_v(x_y + vec2<i32>( 1,  0)).x;

  let up_left    = read_u_v(x_y + vec2<i32>(-1, -1)).x;
  let up_right   = read_u_v(x_y + vec2<i32>( 1, -1)).x;
  let down_left  = read_u_v(x_y + vec2<i32>(-1,  1)).x;
  let down_right = read_u_v(x_y + vec2<i32>( 1,  1)).x;

  return nine_point_stencil_u(
      center_u, up, down, left, right,
      up_left, up_right, down_left, down_right
  );
}

// Single-pass Forward Euler step
@compute @workgroup_size(16, 16)
fn main_euler(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(tex_in);
  let dt = u_time.dt;

  if (gid.x >= dims.x || gid.y >= dims.y) {
      return;
  }

  let x_y = vec2<i32>(i32(gid.x), i32(gid.y));

  let u_v = read_u_v(x_y);
  let u = u_v.x;
  let v = u_v.y;

  let lap_u = laplacian_u(u, x_y);
  let rhs = fhn_system_rhs(u, v, lap_u);

  // Forward Euler update: y_{n+1} = y_n + dt * f(y_n)
  let u_next = u + rhs.x * dt;
  let v_next = v + rhs.y * dt;

  textureStore(tex_out, x_y, vec4<f32>(u_next, v_next, 0.0, 1.0));
}
