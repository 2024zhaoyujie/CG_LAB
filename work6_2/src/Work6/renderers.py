from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
)


def build_silhouette_renderer(cameras, image_size: int, sigma: float, faces_per_pixel: int):
    blend = BlendParams(sigma=sigma, gamma=sigma)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend),
    )


def build_rgb_renderer(cameras, image_size: int, sigma: float, faces_per_pixel: int, device):
    blend = BlendParams(sigma=sigma, gamma=sigma)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
    )
    lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend),
    )
