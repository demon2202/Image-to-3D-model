"""
Multi-view depth fusion using TSDF volume.
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional


class TSDFVolume:
    """
    Truncated Signed Distance Function volume for depth fusion.
    Uses Open3D's ScalableTSDFVolume for efficiency.
    """
    
    def __init__(
        self,
        voxel_length: float = 0.005,
        sdf_trunc: float = 0.04,
        color_type: int = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    ):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=color_type
        )
    
    def integrate(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        depth_scale: float = 1000.0,
        depth_trunc: float = 5.0
    ):
        """
        Integrate a color + depth frame into the volume.
        """
        # Create Open3D images
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) 
                                        if len(color_image.shape) == 3 else color_image)
        depth_o3d = o3d.geometry.Image((depth_image * depth_scale).astype(np.uint16))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # Camera pose as 4x4 matrix (world-to-camera -> camera-to-world for Open3D)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t.ravel()
        camera_to_world = np.linalg.inv(extrinsic)
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=color_image.shape[1],
            height=color_image.shape[0],
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2]
        )
        
        self.volume.integrate(rgbd, intrinsic, camera_to_world)
    
    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        """Extract point cloud from TSDF volume."""
        return self.volume.extract_point_cloud()
    
    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """Extract triangle mesh from TSDF volume."""
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh


def fuse_depth_maps(
    images: List[np.ndarray],
    depth_maps: List[np.ndarray],
    camera_poses: List[Tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    voxel_length: float = 0.005
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """
    Fuse multiple depth maps into a consistent 3D model.
    
    Returns:
        (point_cloud, mesh)
    """
    import cv2  # Import here to avoid circular imports
    
    tsdf = TSDFVolume(voxel_length=voxel_length)
    
    for i, (img, depth, (R, t)) in enumerate(zip(images, depth_maps, camera_poses)):
        print(f"  Integrating frame {i+1}/{len(images)}")
        tsdf.integrate(img, depth, K, R, t)
    
    pcd = tsdf.extract_point_cloud()
    mesh = tsdf.extract_mesh()
    
    return pcd, mesh