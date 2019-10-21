extern crate image;
extern crate minifb;
extern crate rand;

use minifb::{Key, Window, WindowOptions};
use rand::Rng;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;
use std::path::Path;
use std::time::Instant;

const RESOLUTION: usize = 768;

#[derive(Copy, Clone, Debug)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}
#[derive(Copy, Clone, Debug)]
struct Vector4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[derive(Copy, Clone)]
struct Matrix4 {
    data: [[f32; 4]; 4],
}

struct Rays {
    center: Vector3,
    color: Vector3,
    coords: Vec<Vector3>,
    world_mtx: Matrix4,
}

const ZERO_VECTOR: Vector3 = Vector3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

impl Vector3 {
    fn new(vx: f32, vy: f32, vz: f32) -> Vector3 {
        Vector3 {
            x: vx,
            y: vy,
            z: vz,
        }
    }
}

impl Sub for Vector3 {
    type Output = Vector3;
    fn sub(self: Vector3, b: Vector3) -> Vector3 {
        Vector3 {
            x: self.x - b.x,
            y: self.y - b.y,
            z: self.z - b.z,
        }
    }
}
impl Add for Vector3 {
    type Output = Vector3;
    fn add(self: Vector3, b: Vector3) -> Vector3 {
        Vector3 {
            x: self.x + b.x,
            y: self.y + b.y,
            z: self.z + b.z,
        }
    }
}
impl Div<f32> for Vector3 {
    type Output = Vector3;
    fn div(self: Vector3, s: f32) -> Vector3 {
        Vector3 {
            x: self.x / s,
            y: self.y / s,
            z: self.z / s,
        }
    }
}
impl Mul<f32> for Vector3 {
    type Output = Vector3;
    fn mul(self: Vector3, s: f32) -> Vector3 {
        Vector3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Matrix4 {
    fn new(
        m00: f32,
        m01: f32,
        m02: f32,
        m03: f32,
        m10: f32,
        m11: f32,
        m12: f32,
        m13: f32,
        m20: f32,
        m21: f32,
        m22: f32,
        m23: f32,
        m30: f32,
        m31: f32,
        m32: f32,
        m33: f32,
    ) -> Matrix4 {
        Matrix4 {
            data: [
                [m00, m01, m02, m03],
                [m10, m11, m12, m13],
                [m20, m21, m22, m23],
                [m30, m31, m32, m33],
            ],
        }
    }
}

macro_rules! mtx_elem {
    ($a : ident, $b : ident, $r : expr, $c : expr) => {
        $a.data[$r][0] * $b.data[0][$c]
            + $a.data[$r][1] * $b.data[1][$c]
            + $a.data[$r][2] * $b.data[2][$c]
            + $a.data[$r][3] * $b.data[3][$c]
    };
}

impl Mul for Matrix4 {
    type Output = Matrix4;
    fn mul(self: Matrix4, b: Matrix4) -> Matrix4 {
        Matrix4::new(
            mtx_elem!(self, b, 0, 0),
            mtx_elem!(self, b, 0, 1),
            mtx_elem!(self, b, 0, 2),
            mtx_elem!(self, b, 0, 3),
            mtx_elem!(self, b, 1, 0),
            mtx_elem!(self, b, 1, 1),
            mtx_elem!(self, b, 1, 2),
            mtx_elem!(self, b, 1, 3),
            mtx_elem!(self, b, 2, 0),
            mtx_elem!(self, b, 2, 1),
            mtx_elem!(self, b, 2, 2),
            mtx_elem!(self, b, 2, 3),
            mtx_elem!(self, b, 3, 0),
            mtx_elem!(self, b, 3, 1),
            mtx_elem!(self, b, 3, 2),
            mtx_elem!(self, b, 3, 3),
        )
    }
}

fn transform_point4(v: &Vector3, m: &Matrix4) -> Vector4 {
    Vector4 {
        x: v.x * m.data[0][0] + v.y * m.data[1][0] + v.z * m.data[2][0] + m.data[3][0],
        y: v.x * m.data[0][1] + v.y * m.data[1][1] + v.z * m.data[2][1] + m.data[3][1],
        z: v.x * m.data[0][2] + v.y * m.data[1][2] + v.z * m.data[2][2] + m.data[3][2],
        w: v.x * m.data[0][3] + v.y * m.data[1][3] + v.z * m.data[2][3] + m.data[3][3],
    }
}

fn perspective_projection_matrix(
    vertical_fov_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> Matrix4 {
    let tfov = (vertical_fov_deg.to_radians() / 2.0).tan();
    let m00 = 1.0 / (aspect * tfov);
    let m11 = 1.0 / tfov;
    let oo_fmn = 1.0 / (far - near);
    let m22 = -(far + near) * oo_fmn;
    let m23 = -(2.0 * far * near) * oo_fmn;

    Matrix4::new(
        m00, 0.0, 0.0, 0.0, 0.0, m11, 0.0, 0.0, 0.0, 0.0, m22, m23, 0.0, 0.0, -1.0, 0.0,
    )
}

fn view_matrix(cam_dist: f32) -> Matrix4 {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -cam_dist, 1.0,
    )
}

fn rot_matrix(angle_degrees: f32, axis: &Vector3) -> Matrix4 {
    // Assumes normalized axis

    let angle_rads = angle_degrees.to_radians();
    let s = angle_rads.sin();
    let x = s * axis.x;
    let y = s * axis.y;
    let z = s * axis.z;
    let w = angle_rads.cos();

    matrix_from_quat(x, y, z, w)
}

fn matrix_from_quat(qx: f32, qy: f32, qz: f32, qw: f32) -> Matrix4 {
    let mut xx = 2.0 * qx;
    let mut yy = 2.0 * qy;
    let mut zz = 2.0 * qz;
    let xxy = xx * qy;
    let xxz = xx * qz;
    let yyz = yy * qz;
    let xxw = xx * qw;
    let yyw = yy * qw;
    let zzw = zz * qw;

    xx *= qx;
    yy *= qy;
    zz *= qz;

    Matrix4::new(
        1.0 - yy - zz,
        xxy + zzw,
        xxz - yyw,
        0.0,
        xxy - zzw,
        1.0 - yy - zz,
        yyz + xxw,
        0.0,
        xxz + yyw,
        yyz - xxw,
        1.0 - xx - yy,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

fn add_pixel(x: i32, y: i32, z: i32, c: u32, buffer: &mut [u32], depth_buffer: &[i32]) {
    let offset = x as usize + (y as usize * RESOLUTION);
    if z > depth_buffer[offset] {
        let pix = (c & 0xFEFE_FEFF) + (buffer[offset] & 0xFEFE_FEFF);
        let oflow = pix & 0x101_0100;
        let oflow = oflow - (oflow >> 8);
        buffer[offset] = 0xFF00_0000 | oflow | pix;
    }
}

fn draw_line(
    mut x0: f32,
    mut y0: f32,
    mut z0: f32,
    mut x1: f32,
    mut y1: f32,
    mut z1: f32,
    r: u32,
    g: u32,
    b: u32,
    buffer: &mut [u32],
    depth_buffer: &[i32],
) {
    // Calc deltas
    let mut delta_x = x1 - x0;
    let mut delta_y = y1 - y0;
    let mut delta_z = z1 - z0;

    // Combine to base color
    let color = (r << 16) | (g << 8) | b;
    let mut temp : f32;
    if delta_x.abs() > delta_y.abs() {
        // Swap coords if needed (delta always positive!)
        if x0 > x1 {
            temp = x0;
            x0 = x1;
            x1 = temp;
            temp = y0;
            y0 = y1;
            y1 = temp;
            temp = z0;
            z0 = z1;
            z1 = temp;
            delta_x = -delta_x;
            delta_y = -delta_y;
            delta_z = -delta_z;
        }

        let mut ix0 = x0.ceil() as i32;

        // First pixel
        add_pixel(
            ix0,
            y0.ceil() as i32,
            (65536.0 * z0) as i32,
            color,
            buffer,
            depth_buffer,
        );

        // Calc line gradients (fixed points)
        let grad_y = (65536.0 * delta_y / delta_x) as i32;
        let grad_z = (65536.0 * delta_z / delta_x) as i32;

        let prestep = ((x0.ceil() - x0) * 1.0) as i32;
        let mut y = (y0 * 65536.0) as i32 + (prestep * grad_y);
        let mut z = (z0 * 65536.0) as i32 + (prestep * grad_z);

        let mut count = x1.ceil() as i32 - ix0 - 1;
        while count > 0 {
            ix0 += 1;
            y += grad_y;
            z += grad_z;

            // Calc brightness
            let br2 = ((y & 0xFFFF) >> 8) as u32;
            let br1 = 256 - br2;
            let r1 = (br1 * r) >> 8;
            let g1 = (br1 * g) >> 8;
            let b1 = (br1 * b) >> 8;
            let r2 = (br2 * r) >> 8;
            let g2 = (br2 * g) >> 8;
            let b2 = (br2 * b) >> 8;

            let iy = y >> 16;
            add_pixel(
                ix0,
                iy,
                z,
                (r1 << 16) | (g1 << 8) | b1,
                buffer,
                depth_buffer,
            );
            add_pixel(
                ix0,
                iy + 1,
                z,
                (r2 << 16) | (g2 << 8) | b2,
                buffer,
                depth_buffer,
            );

            count -= 1;
        }

        // Final pixel
        add_pixel(
            x1.ceil() as i32,
            y1.ceil() as i32,
            (65536.0 * z1) as i32,
            color,
            buffer,
            depth_buffer,
        );
    } 
    else {
        // Swap coords if needed (delta always positive!)
        if y0 > y1 {
            temp = x0;
            x0 = x1;
            x1 = temp;
            temp = y0;
            y0 = y1;
            y1 = temp;
            temp = z0;
            z0 = z1;
            z1 = temp;
            delta_x = -delta_x;
            delta_y = -delta_y;
            delta_z = -delta_z;
        }

        let mut iy0 = y0.ceil() as i32;
        //System.out.println(iy0);

        // First pixel
        add_pixel(
            x0.ceil() as i32,
            iy0,
            (65536.0 * z0) as i32,
            color,
            buffer,
            depth_buffer,
        );

        // Calc line gradients (fixed points)
        let grad_x = (65536.0 * delta_x / delta_y) as i32;
        let grad_z = (65536.0 * delta_z / delta_y) as i32;

        let prestep = ((y0.ceil() - y0) * 1.0) as i32;
        let mut x = (x0 * 65536.0) as i32 + (prestep * grad_x);
        let mut z = (z0 * 65536.0) as i32 + (prestep * grad_z);

        let mut count = y1.ceil() as i32 - iy0 - 1;
        while count > 0 {
            x += grad_x;
            iy0 += 1;
            z += grad_z;

            count -= 1;

            // Calc brightness
            let br2 = ((x & 0xFFFF) >> 8) as u32;
            let br1 = 256 - br2;
            let r1 = (br1 * r) >> 8;
            let g1 = (br1 * g) >> 8;
            let b1 = (br1 * b) >> 8;
            let r2 = (br2 * r) >> 8;
            let g2 = (br2 * g) >> 8;
            let b2 = (br2 * b) >> 8;

            let ix = x >> 16;
            add_pixel(
                ix,
                iy0,
                z,
                (r1 << 16) | (g1 << 8) | b1,
                buffer,
                depth_buffer,
            );
            add_pixel(
                ix + 1,
                iy0,
                z,
                (r2 << 16) | (g2 << 8) | b2,
                buffer,
                depth_buffer,
            );
        }

        // Final pixel
        add_pixel(
            x1.ceil() as i32,
            y1.ceil() as i32,
            (65536.0 * z1) as i32,
            color,
            buffer,
            depth_buffer,
        );
    }
}

fn blur(buffer: &mut [u32]) {
    let w = RESOLUTION;
    let h = RESOLUTION;

    for y in h - 2..1 {
        for x in w - 2..1 {
            let ofs = y * w + x;

            //                m_backBuffer[ofs] = ((m_backBuffer[ofs - 1] >> 1) & 0x7F7F7F7F) +
            //                                    ((m_backBuffer[ofs + 1] >> 1) & 0x7F7F7F7F);
            buffer[ofs] = ((buffer[ofs - 1] >> 2) & 0x3F3F_3F3F)
                + ((buffer[ofs + 1] >> 2) & 0x3F3F_3F3F)
                + ((buffer[ofs + w] >> 2) & 0x3F3F_3F3F)
                + ((buffer[ofs - w] >> 2) & 0x3F3F_3F3F);
        }
    }
}

///
// Prepares depth buffer (if you don't write to it then it's enough to call
// it only one time).
// Algorithm is very simple: it scans occlusion image and:
//  * if current pixel is lit then quite "near" value is written
//    (for example, writing 1/700 would mean that the plane
//    containing occluder lies 700 units from the camera). Only few rays
//    will overwrite pixel at this location.
//  * if current pixel is black then infinitely "far" value is written.
//    That way every ray will easily overwrite this pixel
//
// Please note that each entry is in 16:16 fixed-point format (well, I use
// only 1/W values, so most probably some 8:24 or 0:31 would be better...)

// 1/W is scaled a bit, in order to increase the precision of depth-buffer
const W_SCALER: f32 = 4096.0;
fn prepare_depth_buffer(
    camera_dist: f32,
    plane_dist: f32,
    occlusion_buffer: &[u8],
    depth_buffer: &mut [i32],
) {
    // Calc plane dist in camera space (inverted)
    let pdist = 1.0 / (camera_dist - plane_dist);

    assert!(depth_buffer.len() == occlusion_buffer.len() / 3);
    for (i, it_depth) in depth_buffer.iter_mut().enumerate() {
        let j = i * 3;
        //let pix = occlusion_buffer[j];
        //let crit = ((pix >> 16) & 0xFF) + ((pix >> 8) & 0xFF) + (pix & 0xFF);
        let crit = occlusion_buffer[j] as u32
            + occlusion_buffer[j + 1] as u32
            + occlusion_buffer[j + 2] as u32;

        if crit > 0x3F {
            *it_depth = (pdist * 65536.0 * W_SCALER) as i32;
        } else {
            *it_depth = 0;
        }
    }
}

fn init_rays(config: &[usize], rays: &mut Rays) {
    let num_stages = config[0];
    let mut num_rays = 0;

    for s in 0..num_stages {
        num_rays += config[1 + s * 2] * 2;
    }

    rays.coords = vec![ZERO_VECTOR; num_rays];
    let mut cur_ray = 0;
    let mut center = ZERO_VECTOR;

    let mut rng = rand::thread_rng();

    // The following code (generating endpoints for the rays) is taken
    // almost as-is from sources of Swop 64k intro by Proxium
    // (file !galaxy.h), should be available for download at any decent
    // scene server.
    for k in 0..num_stages {
        let radius = config[2 + k * 2] as f32;
        println!("Radius: {}", radius);
        for _i in 0..config[1 + k * 2] * 2 {
            let r = rng.gen::<f32>() * radius;
            let phi = (std::f32::consts::PI / 180.0) * rng.gen::<f32>() * 359.0;
            let psi = (std::f32::consts::PI / 180.0) * rng.gen::<f32>() * 359.0;

            rays.coords[cur_ray] = Vector3::new(
                r * phi.cos(),
                r * (phi.sin() * psi.cos()),
                r * (phi.sin() * psi.sin()),
            );

            center = center + rays.coords[cur_ray];

            cur_ray += 1;
        }
    }

    // Calc center
    center = center / (cur_ray as f32);

    // Now translate all the points so that center lies at (0, 0, 0)
    // in local space.
    rays.coords.iter_mut().for_each(|v| *v = *v - center);
    rays.center = center;
}

// Hope this gets inlined!!
//
// Returns true if point is inside view volume
fn transform(vout: &mut Vector4, v: &Vector3, m: &Matrix4) -> bool {
    //vout.transform(v, m);
    *vout = transform_point4(&v, &m);

    (vout.x > -vout.w && vout.x < vout.w)
    && (vout.y > -vout.w && vout.y < vout.w)
    && (vout.z > 0.0 && vout.z < vout.w)
}

fn transform_line_point(
    vout: &mut Vector4,
    v: &Vector3,
    m: &Matrix4,
    transx2d: f32,
    cx: &mut f32,
    cy: &mut f32,
    oow: &mut f32,
) -> bool {
    if !transform(vout, v, m) {
        return false;
    }
    *oow = 1.0 / vout.w;
    // 126 (anti-aliasing!)
    const AA_SCALE: f32 = (RESOLUTION / 2) as f32 - 2.0;
    const MAX_D: f32 = RESOLUTION as f32 - 4.0;
    const MIN_D: f32 = 4.0;
    *cx = AA_SCALE + (AA_SCALE * vout.x * *oow) + transx2d;
    if *cx > MAX_D {
        *cx = MAX_D;
    } else if *cx < MIN_D {
        *cx = MIN_D;
    }
    *cy = AA_SCALE - (AA_SCALE * vout.y * *oow);
    // Scale 1/w
    *oow *= W_SCALER;

    true
}

fn transform_and_draw(
    rays: &Rays,
    num_rays: usize,
    transx2d: f32,
    viewproj: &Matrix4,
    buffer: &mut [u32],
    depth_buffer: &[i32],
) {
    let proj_matrix = rays.world_mtx * *viewproj;

    // Center point (return if not visible.. this is not very correct, but
    // I don't support clipping, so...)
    let mut vtmp = Vector4 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut coow = 0.0;
    if !transform_line_point(
        &mut vtmp,
        &rays.center,
        &proj_matrix,
        transx2d,
        &mut cx,
        &mut cy,
        &mut coow,
    ) {
        return;
    }

    let r = rays.color.x as u32;
    let g = rays.color.y as u32;
    let b = rays.color.z as u32;

    for i in 0..num_rays {
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut oow = 0.0;
        if !transform_line_point(
            &mut vtmp,
            &rays.coords[i],
            &proj_matrix,
            transx2d,
            &mut sx,
            &mut sy,
            &mut oow,
        ) {
            continue;
        }

        draw_line(cx, cy, coow, sx, sy, oow, r, g, b, buffer, depth_buffer);
    }
}

fn clear_buffer(buffer: &mut [u32]) {
    for it in buffer.iter_mut().take(RESOLUTION * RESOLUTION) {
        *it = 0;
    }
}

fn main() {
    let mut window = Window::new(
        "rust_light - ESC to exit",
        RESOLUTION,
        RESOLUTION,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let img = image::open(&Path::new("starlight1.jpg")).expect("Opening image failed");
    let img_rgb = img.as_rgb8().unwrap();
    let texture_pixels = img_rgb.clone().into_raw();

    let mut framebuffer: Vec<u32> = vec![0; RESOLUTION * RESOLUTION];
    let mut depth_buffer: Vec<i32> = vec![0; RESOLUTION * RESOLUTION];

    let config: [usize; 1 + 2 * 4] = [
        4, 100, 350, // Stage #0
        140, 900, // Stage #1
        250, 1400, // Stage #2
        180, 2250,
    ]; // Stage #3

    let light_rot_axis = Vector3::new(1.0, 1.0, 0.0);
    let mut angle = 0.0;
    let mut rays = Rays {
        center: ZERO_VECTOR,
        color: Vector3::new(20.0, 63.0, 127.0),
        coords: vec![],
        world_mtx: rot_matrix(angle, &light_rot_axis),
    };
    init_rays(&config, &mut rays);

    let camera_dist = 1200.0;
    let plane_dist = 120.0;

    let world_to_cam = view_matrix(camera_dist);
    let proj_matrix = perspective_projection_matrix(90.0, 1.0, 1.0, 5000.0);
    let viewproj = world_to_cam * proj_matrix;

    prepare_depth_buffer(
        camera_dist,
        plane_dist,
        texture_pixels.as_slice(),
        depth_buffer.as_mut_slice(),
    );

    let mut xpos = 0.0;
    let num_rays = rays.coords.len();
    let app_start_time = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let start_time = Instant::now();

        clear_buffer(framebuffer.as_mut_slice());
        transform_and_draw(
            &rays,
            num_rays,
            xpos,
            &viewproj,
            framebuffer.as_mut_slice(),
            depth_buffer.as_slice(),
        );
        blur(framebuffer.as_mut_slice());
        window.update_with_buffer(framebuffer.as_slice()).unwrap();

        let time_taken = Instant::now().duration_since(start_time);
        let time_taken_dbl = time_taken.as_secs() as f64 + f64::from(time_taken.subsec_nanos()) * 1e-9;
        let fps = (1.0 / time_taken_dbl) as u32;
        window.set_title(&format!("FPS {}", fps));

        let total_time = Instant::now().duration_since(app_start_time).as_millis() as f32;
        angle = (total_time / 45.0) % 360.0;
        rays.world_mtx = rot_matrix(angle, &light_rot_axis);

        // THIS SUCKS. Hopefully won't run too strange on fast computers...
        // Translation is done in 2D to make it independent from the
        // camera parameters
        xpos = 90.0 * (total_time / 20_000.0).sin();
    }
}
