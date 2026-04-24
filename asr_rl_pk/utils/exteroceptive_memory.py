import numpy as np
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt

# =========================
# Basic SE(3) utilities
# =========================

def make_transform(R: np.ndarray, t: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Build 4x4 homogeneous transform from R(3,3), t(3,)
    """
    T = np.eye(4, dtype=dtype)
    T[:3, :3] = R.astype(dtype)
    T[:3, 3] = t.astype(dtype)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert 4x4 SE(3) transform.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compose two 4x4 transforms.
    Returns A @ B
    """
    return A @ B


def transform_points(points: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    """
    Transform Nx3 points with 4x4 transform.

    points: (N, 3)
    T_dst_src: 4x4, point in src -> point in dst
    """
    if points.size == 0:
        return points.copy()

    R = T_dst_src[:3, :3]
    t = T_dst_src[:3, 3]
    return points @ R.T + t


# =========================
# Camera model
# =========================

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class DepthProjector:
    """
    Precompute pixel rays so repeated depth->pointcloud is faster.
    Assumes depth image stores camera-frame z-depth.
    """

    def __init__(self, intr: CameraIntrinsics, stride: int = 1, dtype=np.float32):
        self.intr = intr
        self.stride = stride
        self.dtype = dtype

        v, u = np.mgrid[0:intr.height:stride, 0:intr.width:stride]
        u = u.astype(dtype)
        v = v.astype(dtype)

        # normalized image plane coordinates
        self.x_factor = (u - intr.cx) / intr.fx
        self.y_factor = (v - intr.cy) / intr.fy

        self.out_h, self.out_w = self.x_factor.shape

    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        depth_min: float = 0.05,
        depth_max: float = 5.0,
        valid_mask: np.ndarray | None = None,
        return_masked_uv: bool = False,
    ):
        """
        Convert z-depth image to Nx3 point cloud in camera frame.

        depth: (H, W), z-depth in meters
        return:
            points_cam: (N, 3)
            optionally u_valid, v_valid for debugging
        """
        assert depth.ndim == 2, "depth must be HxW"
        assert depth.shape == (self.intr.height, self.intr.width), \
            f"depth shape must be {(self.intr.height, self.intr.width)}"

        depth_ds = depth[::self.stride, ::self.stride].astype(self.dtype, copy=False)

        mask = np.isfinite(depth_ds)
        mask &= (depth_ds >= depth_min)
        mask &= (depth_ds <= depth_max)

        if valid_mask is not None:
            vm = valid_mask[::self.stride, ::self.stride]
            mask &= vm.astype(bool)

        Z = depth_ds[mask]
        X = self.x_factor[mask] * Z
        Y = self.y_factor[mask] * Z

        points_cam = np.stack((X, Y, Z), axis=1).astype(self.dtype, copy=False)

        if return_masked_uv:
            vv, uu = np.mgrid[0:self.intr.height:self.stride, 0:self.intr.width:self.stride]
            return points_cam, uu[mask], vv[mask]

        return points_cam


# =========================
# Optional point filtering / sampling
# =========================

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Lightweight NumPy voxel downsampling.
    Keep first point in each voxel.
    """
    if points.size == 0 or voxel_size <= 0:
        return points

    coords = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx]


def random_sample_points(points: np.ndarray, max_points: int, rng=None) -> np.ndarray:
    """
    Randomly sample up to max_points.
    """
    if points.shape[0] <= max_points:
        return points
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


# =========================
# Exteroceptive memory
# =========================

@dataclass
class CloudFrame:
    points_robot: np.ndarray      # point cloud already in robot frame at capture time
    T_world_robot: np.ndarray     # robot pose in world at capture time
    timestamp: float | None = None


class ExteroceptiveMemoryBuffer:
    """
    Store last N point clouds.
    Each cloud is stored in robot frame at capture time.
    When querying, old clouds are transformed into current robot frame.

    This matches the common exteroceptive memory idea:
    old pointclouds -> aligned to current body frame -> concatenate.
    """

    def __init__(
        self,
        max_frames: int,
        max_points_per_frame: int | None = None,
        voxel_size: float | None = None,
        dtype=np.float32,
    ):
        self.max_frames = max_frames
        self.max_points_per_frame = max_points_per_frame
        self.voxel_size = voxel_size
        self.dtype = dtype
        self.buffer = deque(maxlen=max_frames)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def push_cloud_robot(
        self,
        points_robot: np.ndarray,
        T_world_robot: np.ndarray,
        timestamp: float | None = None,
    ):
        """
        Push a cloud already expressed in robot frame.
        """
        pts = np.asarray(points_robot, dtype=self.dtype)

        if self.voxel_size is not None:
            pts = voxel_downsample(pts, self.voxel_size)

        if self.max_points_per_frame is not None:
            pts = random_sample_points(pts, self.max_points_per_frame)

        frame = CloudFrame(
            points_robot=pts,
            T_world_robot=np.asarray(T_world_robot, dtype=self.dtype),
            timestamp=timestamp,
        )
        self.buffer.append(frame)

    def push_depth(
        self,
        depth: np.ndarray,
        projector: DepthProjector,
        T_robot_cam: np.ndarray,
        T_world_robot: np.ndarray,
        depth_min: float = 0.05,
        depth_max: float = 5.0,
        valid_mask: np.ndarray | None = None,
        timestamp: float | None = None,
    ):
        """
        Convert depth -> camera cloud -> robot cloud, then store.
        當前這一幀 depth image → camera frame 點雲 → robot frame 點雲 → 存進 buffer
        """
        points_cam = projector.depth_to_pointcloud(
            depth=depth,
            depth_min=depth_min,
            depth_max=depth_max,
            valid_mask=valid_mask,
        )
        points_robot = transform_points(points_cam, T_robot_cam)
        self.push_cloud_robot(points_robot, T_world_robot, timestamp=timestamp)

    def get_merged_cloud_in_current_robot(
        self,
        T_world_robot_current: np.ndarray,
        max_total_points: int | None = None,
        newest_first: bool = False,
    ) -> np.ndarray:
        """
        Transform all stored robot-frame clouds to current robot frame and concatenate.

        old cloud in robot_old frame:
            p_robot_now = T_robot_now_robot_old * p_robot_old

        where:
            T_robot_now_robot_old = inv(T_world_robot_now) @ T_world_robot_old
        """
        if len(self.buffer) == 0:
            return np.zeros((0, 3), dtype=self.dtype)

        T_robot_now_world = invert_transform(np.asarray(T_world_robot_current, dtype=self.dtype))

        frames = list(self.buffer)
        if newest_first:
            frames = frames[::-1]

        merged = []
        for frame in frames:
            T_robot_now_robot_old = compose_transform(T_robot_now_world, frame.T_world_robot)
            pts_now = transform_points(frame.points_robot, T_robot_now_robot_old)
            merged.append(pts_now)

        merged = np.concatenate(merged, axis=0)

        if max_total_points is not None:
            merged = random_sample_points(merged, max_total_points)

        return merged.astype(self.dtype, copy=False)

    def get_clouds_in_current_robot(
        self,
        T_world_robot_current: np.ndarray,
        newest_first: bool = False,
    ) -> list[np.ndarray]:
        """
        Return list of aligned clouds, one array per stored frame.
        """
        if len(self.buffer) == 0:
            return []

        T_robot_now_world = invert_transform(np.asarray(T_world_robot_current, dtype=self.dtype))

        frames = list(self.buffer)
        if newest_first:
            frames = frames[::-1]

        out = []
        for frame in frames:
            T_robot_now_robot_old = compose_transform(T_robot_now_world, frame.T_world_robot)
            pts_now = transform_points(frame.points_robot, T_robot_now_robot_old)
            out.append(pts_now)

        return out
    
    def debug_show_points(
        self,
        points: np.ndarray,
        title: str = "Point Cloud",
        elev: float = 25,
        azim: float = -60,
        s: float = 1.0,
    ):
        """
        Show one point cloud.
        """
        if points.shape[0] == 0:
            print("No points.")
            return

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=s,
            c=points[:, 2],
            cmap="jet",
        )

        ax.set_xlabel("X forward")
        ax.set_ylabel("Y left")
        ax.set_zlabel("Z up")
        ax.set_title(title)

        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1,1,0.5])
        plt.tight_layout()
        plt.show()

    def debug_show_buffer(
        self,
        T_world_robot_current: np.ndarray,
        newest_first: bool = False,
    ):
        """
        Show all stored clouds aligned to current robot frame.
        Different frame = different color.
        """
        clouds = self.get_clouds_in_current_robot(
            T_world_robot_current,
            newest_first=newest_first
        )

        if len(clouds) == 0:
            print("Buffer empty.")
            return

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        colors = plt.cm.viridis(np.linspace(0, 1, len(clouds)))

        for i, pts in enumerate(clouds):
            if pts.shape[0] == 0:
                continue
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=1,
                color=colors[i],
                label=f"frame {i}"
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Stored Clouds in Current Robot Frame")
        ax.legend()

        ax.set_box_aspect([1,1,0.5])
        plt.tight_layout()
        plt.show()

    def debug_show_merged(
        self,
        T_world_robot_current: np.ndarray,
        max_total_points: int | None = 30000,
    ):
        """
        Show merged point cloud that encoder receives.
        """
        pts = self.get_merged_cloud_in_current_robot(
            T_world_robot_current,
            max_total_points=max_total_points
        )

        self.debug_show_points(
            pts,
            title="Merged Cloud for Encoder"
        )
    def crop_points_robot(self, points, x_min, x_max, y_min, y_max, z_min, z_max):
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        return points[mask]