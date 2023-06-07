import os

import numpy as np
import rospy
import taichi as ti
from osgeo import gdal
from scene import Scene
from taichi.math import *


# this also rospy.init_node()s
voxel_dx = 0.01
scene = Scene(voxel_dx=voxel_dx, voxel_edges=0.01, exposure=1)
scene.set_directional_light((1, 1, 1), 0.3, (1, 1, 1))
scene.set_background_color((0.3, 0.4, 0.6))
scene.set_floor(-10, (0.4, 0.3, 0.12))

def load_data():

    color_image = os.path.expanduser(rospy.get_param("~color", ""))
    height_image = os.path.expanduser(rospy.get_param("~height", ""))

    rospy.loginfo(f"loading color '{color_image}'")
    color_data = gdal.Open(color_image)
    if color_data is None:
        rospy.logerr(color_image)
        return

    shape = color_data.GetRasterBand(1).ReadAsArray().shape
    color_bands = np.zeros((shape[0], shape[1], 3), np.uint8)
    for i in range(3):
        color_bands[:, :, i] = color_data.GetRasterBand(i + 1).ReadAsArray()

    rospy.loginfo(f"loading height '{height_image}'")
    height_data = gdal.Open(height_image)
    if height_data is None:
        rospy.logerr(height_image)
        return

    ind = 1
    height_raster = height_data.GetRasterBand(ind)
    height_band = height_raster.ReadAsArray()
    height_band[height_band == height_raster.GetNoDataValue()] = np.nan

    return color_data, color_bands, height_data, height_band


color_data, color_bands, height_data, height_band = load_data()
# color_ti = ti.field(int, shape=color_bands.shape)
# color_ti.from_numpy(color_bands)
# print(type(color_ti))
print(color_bands.shape)
print(color_bands.dtype)
print(type(color_bands))


@ti.kernel
def initialize_voxels(color_bands: ti.types.ndarray(), height_band: ti.types.ndarray()):
    x0 = 2180
    # TODO(lucasw) there are gaps if the range gets larger than this
    x1 = x0 + 60
    y0 = 2200
    y1 = y0 + 100

    for yi, xj in ti.ndrange((y0, y1), (x0, x1)):
        zk = height_band[yi, xj] * 20.0
        pos = ivec3(yi - y0, zk, xj - x0)
        sc = 255.0
        r = color_bands[yi, xj, 0] / sc
        g = color_bands[yi, xj, 1] / sc
        b = color_bands[yi, xj, 2] / sc
        color_vec = vec3(r, g, b)
        scene.set_voxel(pos, 1, color_vec)


initialize_voxels(color_bands, height_band)
scene.finish()
