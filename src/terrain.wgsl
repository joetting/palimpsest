// Unconformity terrain shader

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) color:    vec3<f32>,
    @location(2) normal:   vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) color:     vec3<f32>,
    @location(2) normal:    vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos  = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.color     = in.color;
    out.normal    = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Simple directional lighting - sun at an angle
    let sun_dir = normalize(vec3<f32>(0.6, 1.0, 0.4));
    let diffuse = max(dot(in.normal, sun_dir), 0.0);
    let ambient = 0.25;
    let light = ambient + (1.0 - ambient) * diffuse;

    // Atmospheric distance fog - gives geological scale feeling
    let dist = length(in.world_pos - uniforms.camera_pos);
    let fog_start = 120.0;
    let fog_end   = 220.0;
    let fog_t = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);

    // Warm, ancient sky haze - ochre tinted horizon
    let sky_color = vec3<f32>(0.68, 0.60, 0.50);

    let lit_color = in.color * light;
    let final_color = mix(lit_color, sky_color, fog_t * fog_t);

    return vec4<f32>(final_color, 1.0);
}
