import numpy as np
import trimesh

def depth_to_point_cloud(depth_map, fx=1.0, fy=1.0):
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    z = depth_map
    x = (i - w / 2) * z / fx
    y = (j - h / 2) * z / fy
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)

def merge_and_save_point_clouds(depth_maps, output_file):
    all_points = []
    for depth in depth_maps:
        pc = depth_to_point_cloud(depth)
        all_points.append(pc)

    full_pc = np.concatenate(all_points, axis=0)
    mesh = trimesh.points.PointCloud(full_pc).convex_hull
    mesh.export(output_file)