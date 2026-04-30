import time

import taichi as ti


ti.init(arch=ti.gpu)

WIDTH = 1280
HEIGHT = 720
ASPECT = WIDTH / HEIGHT
FOV_DEG = 55.0
MAX_BOUNCES_LIMIT = 5
MAX_SPP_LIMIT = 8
EPSILON = 1e-4
MIRROR_REFLECTANCE = 0.8
GLASS_IOR = 1.5
GLASS_TRANSMISSION = 0.98
PLANE_Y = -1.0

MISS = 0
GROUND_DIFFUSE = 1
GLASS = 2
MIRROR = 3

RED_SPHERE_CENTER = ti.Vector([-1.5, 0.0, 0.0])
MIRROR_SPHERE_CENTER = ti.Vector([1.5, 0.0, 0.0])
SPHERE_RADIUS = 1.0

CAMERA_POS = ti.Vector([0.0, 0.7, 6.0])
BG_TOP = ti.Vector([0.03, 0.10, 0.12])
BG_BOTTOM = ti.Vector([0.0, 0.0, 0.0])

image = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.func
def reflect_dir(inc_dir, normal):
    return (inc_dir - 2.0 * inc_dir.dot(normal) * normal).normalized()


@ti.func
def refract_dir(inc_dir, normal, ior):
    cosi = ti.min(1.0, ti.max(-1.0, inc_dir.dot(normal)))
    etai = 1.0
    etat = ior
    n = normal

    if cosi < 0.0:
        cosi = -cosi
    else:
        n = -normal
        etai = ior
        etat = 1.0

    eta = etai / etat
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    can_refract = 1
    out_dir = reflect_dir(inc_dir, normal)
    if k < 0.0:
        can_refract = 0
    else:
        out_dir = (eta * inc_dir + (eta * cosi - ti.sqrt(k)) * n).normalized()
    return can_refract, out_dir


@ti.func
def fresnel_schlick(inc_dir, normal, ior):
    cos_theta = ti.abs(inc_dir.dot(normal))
    cos_theta = ti.min(cos_theta, 1.0)
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    return r0 + (1.0 - r0) * ((1.0 - cos_theta) ** 5)


@ti.func
def sphere_intersect(ro, rd, center, radius):
    oc = ro - center
    b = oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    h = b * b - c
    t = 1e9
    hit = 0
    if h >= 0.0:
        sqrt_h = ti.sqrt(h)
        t0 = -b - sqrt_h
        t1 = -b + sqrt_h
        if t0 > EPSILON:
            t = t0
            hit = 1
        elif t1 > EPSILON:
            t = t1
            hit = 1
    return hit, t


@ti.func
def scene_intersect(ro, rd):
    hit_any = 0
    closest_t = 1e9
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MISS

    hit, t = sphere_intersect(ro, rd, RED_SPHERE_CENTER, SPHERE_RADIUS)
    if hit == 1 and t < closest_t:
        hit_any = 1
        closest_t = t
        p = ro + rd * t
        hit_n = (p - RED_SPHERE_CENTER).normalized()
        hit_mat = GLASS

    hit, t = sphere_intersect(ro, rd, MIRROR_SPHERE_CENTER, SPHERE_RADIUS)
    if hit == 1 and t < closest_t:
        hit_any = 1
        closest_t = t
        p = ro + rd * t
        hit_n = (p - MIRROR_SPHERE_CENTER).normalized()
        hit_mat = MIRROR

    denom = rd.y
    if ti.abs(denom) > 1e-6:
        t_plane = (PLANE_Y - ro.y) / denom
        if t_plane > EPSILON and t_plane < closest_t:
            hit_any = 1
            closest_t = t_plane
            hit_n = ti.Vector([0.0, 1.0, 0.0])
            hit_mat = GROUND_DIFFUSE

    return hit_any, closest_t, hit_n, hit_mat


@ti.func
def checker_albedo(hit_pos):
    xi = ti.floor(hit_pos.x)
    zi = ti.floor(hit_pos.z)
    parity = int(xi + zi) & 1
    c0 = ti.Vector([0.92, 0.92, 0.92])
    c1 = ti.Vector([0.08, 0.08, 0.08])
    return c0 if parity == 0 else c1


@ti.func
def background_color(rd):
    t = 0.5 * (rd.y + 1.0)
    return BG_BOTTOM * (1.0 - t) + BG_TOP * t


@ti.func
def phong_shade(hit_pos, normal, base_color, ray_dir, light_pos):
    ambient = 0.08 * base_color
    light_vec = light_pos - hit_pos
    light_dist = light_vec.norm()
    light_dir = light_vec / ti.max(light_dist, 1e-6)

    shadow_origin = hit_pos + normal * EPSILON
    in_shadow = 0
    shadow_hit, shadow_t, _, _ = scene_intersect(shadow_origin, light_dir)
    if shadow_hit == 1 and shadow_t < light_dist:
        in_shadow = 1

    shaded = ambient
    if in_shadow == 0:
        diff = ti.max(normal.dot(light_dir), 0.0)
        diffuse = diff * base_color

        view_dir = (-ray_dir).normalized()
        spec_dir = reflect_dir(-light_dir, normal)
        spec = ti.pow(ti.max(view_dir.dot(spec_dir), 0.0), 64.0) * 0.25
        specular = ti.Vector([spec, spec, spec])
        shaded = ambient + diffuse + specular

    return shaded


@ti.kernel
def render(light_pos: ti.types.vector(3, ti.f32), max_bounces: ti.i32, spp: ti.i32):
    fov_scale = ti.tan(0.5 * FOV_DEG * 3.1415926535 / 180.0)
    for i, j in image:
        pixel_color = ti.Vector([0.0, 0.0, 0.0])

        for s in range(MAX_SPP_LIMIT):
            if s >= spp:
                break

            jitter_x = ti.random(ti.f32) - 0.5
            jitter_y = ti.random(ti.f32) - 0.5
            u = (2.0 * ((i + 0.5 + jitter_x) / WIDTH) - 1.0) * ASPECT * fov_scale
            v = (2.0 * ((j + 0.5 + jitter_y) / HEIGHT) - 1.0) * fov_scale

            ray_origin = CAMERA_POS
            ray_dir = ti.Vector([u, v, -1.0]).normalized()

            throughput = ti.Vector([1.0, 1.0, 1.0])
            sample_color = ti.Vector([0.0, 0.0, 0.0])

            for bounce in range(MAX_BOUNCES_LIMIT):
                if bounce >= max_bounces:
                    break

                hit, t, normal, mat_id = scene_intersect(ray_origin, ray_dir)
                if hit == 0:
                    sample_color += throughput * background_color(ray_dir)
                    break

                hit_pos = ray_origin + ray_dir * t

                if mat_id == MIRROR:
                    ray_dir = reflect_dir(ray_dir, normal)
                    origin_offset = normal * EPSILON
                    if ray_dir.dot(normal) < 0.0:
                        origin_offset = -origin_offset
                    ray_origin = hit_pos + origin_offset
                    throughput *= MIRROR_REFLECTANCE
                elif mat_id == GLASS:
                    refl_dir = reflect_dir(ray_dir, normal)
                    can_refract, refr_dir = refract_dir(ray_dir, normal, GLASS_IOR)
                    reflect_prob = fresnel_schlick(ray_dir, normal, GLASS_IOR)

                    out_dir = refl_dir
                    if can_refract == 1 and ti.random(ti.f32) > reflect_prob:
                        out_dir = refr_dir

                    origin_offset = normal * EPSILON
                    if out_dir.dot(normal) < 0.0:
                        origin_offset = -origin_offset
                    ray_origin = hit_pos + origin_offset
                    ray_dir = out_dir
                    throughput *= GLASS_TRANSMISSION
                else:
                    base_color = checker_albedo(hit_pos)
                    local_color = phong_shade(hit_pos, normal, base_color, ray_dir, light_pos)
                    sample_color += throughput * local_color
                    break

            pixel_color += sample_color

        final_color = pixel_color / ti.max(1, spp)
        image[i, j] = ti.min(final_color, ti.Vector([1.0, 1.0, 1.0]))


def main():
    window = ti.ui.Window("Whitted Ray Tracing Lab", (WIDTH, HEIGHT), fps_limit=60)
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x = 0.1
    light_y = 5.0
    light_z = 2.0
    max_bounces = 3
    spp = 4

    last_time = time.perf_counter()
    fps = 0.0

    while window.running:
        now = time.perf_counter()
        dt = max(now - last_time, 1e-6)
        last_time = now
        fps = 1.0 / dt

        gui.begin("Controls", 0.02, 0.70, 0.28, 0.28)
        light_x = gui.slider_float("Light X", light_x, -6.0, 6.0)
        light_y = gui.slider_float("Light Y", light_y, 0.2, 8.0)
        light_z = gui.slider_float("Light Z", light_z, -6.0, 6.0)
        max_bounces = gui.slider_int("Max Bounces", max_bounces, 1, 5)
        spp = gui.slider_int("MSAA Samples", spp, 1, MAX_SPP_LIMIT)
        gui.text(f"Glass IOR = {GLASS_IOR:.2f}")
        gui.text(f"Mirror reflectance = {MIRROR_REFLECTANCE:.1f}")
        gui.text(f"Epsilon offset = {EPSILON:.0e}")
        gui.text(f"{fps:.2f} FPS")
        gui.end()

        render(ti.Vector([light_x, light_y, light_z]), max_bounces, spp)
        canvas.set_image(image)
        window.show()


if __name__ == "__main__":
    main()
