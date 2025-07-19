import math
from functools import lru_cache

import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F

from scene import GaussianModel
from scene.cameras import Camera


def unproject(depth_map: torch.Tensor, extrinsic_matrix: torch.Tensor, intrinsic_matrix: torch.Tensor):
    """
    Args:
        depth_map: (1, H, W)
        extrinsic_matrix: (4, 4) world-to-camera
        intrinsic_matrix: (3, 3) camera-to-pixel
    Returns:
        (3, H, W)
    """
    # assert len(depth_map.shape) == 3 and depth_map.shape[0] == 1
    H, W = depth_map.shape[1:]
    depth_map = depth_map.transpose(1, 2)   # torch.Size([1, 400, 400])

    if not isinstance(extrinsic_matrix, torch.Tensor):
        extrinsic_matrix = torch.tensor(extrinsic_matrix, dtype=depth_map.dtype, device=depth_map.device)
    if not isinstance(intrinsic_matrix, torch.Tensor):
        intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=depth_map.dtype, device=depth_map.device)

    u, v = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing="ij")
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=0).float().to(depth_map.device)
    uv1 = uv1.view(3, -1)
    xyz_cam = torch.inverse(intrinsic_matrix) @ uv1 * depth_map.reshape(1, -1)
    xyz1_cam = torch.cat([xyz_cam, torch.ones_like(xyz_cam[:1])], dim=0)

    xyz1_world = torch.inverse(extrinsic_matrix) @ xyz1_cam
    xyz1_world = xyz1_world.view(4, W, H).transpose(1, 2)
    xyz_world = xyz1_world[:3, :, :]    # torch.Size([3, 400, 400])

    return xyz_world


@lru_cache(maxsize=1)
def get_faiss_index():
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="faiss.contrib.torch_utils")

    import faiss
    from faiss.contrib import torch_utils

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(3)
    # index_flat = faiss.IndexIVFFlat(index_flat, 3, 100, faiss.METRIC_L2)
    # index_flat.train(xyz2)
    gpu_index_flat: faiss.IndexFlatL2 = faiss.index_cpu_to_gpu(res, 0, index_flat)

    return gpu_index_flat


def knn(xyz1: torch.Tensor, xyz2: torch.Tensor, K: int, backend="mmcv") -> torch.Tensor:
    """
    Args:
        xyz1: (N, 3)
        xyz2: (M, 3)
        K: int <= M
    Returns:
        (N, K)
    """
    # assert K > 0
    # assert xyz2.shape[0] > K, "K is too large"
    N = xyz1.shape[0]

    if backend in ("o3d", "open3d"):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz2.cpu().numpy(), np.float64))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        top_k_nearest_idx = np.zeros([N, K], dtype=np.int64)
        for i, p in enumerate(xyz1.cpu().numpy()):
            k, idx, d = pcd_tree.search_knn_vector_3d(p, K)
            top_k_nearest_idx[i, :] = idx
        top_k_nearest_idx = torch.Tensor(top_k_nearest_idx).to(xyz1.device).long()
    elif backend == "faiss":
        faiss_index = get_faiss_index()
        faiss_index.add(xyz2)
        d, top_k_nearest_idx = faiss_index.search(xyz1.contiguous(), K)
        faiss_index.reset()
    elif backend == "mmcv":
        from mmcv.ops.knn import knn

        top_k_nearest_idx = knn(K, xyz2.unsqueeze(0).contiguous(), xyz1.unsqueeze(0).contiguous()).squeeze(0).T
        # top_k_nearest_idx = top_k_nearest_idx.long()
    else:
        CHUNK = 2048  # prevent OOM
        top_k_nearest_idx = torch.zeros(N, K, dtype=torch.int64, device=xyz1.device)  # N,K
        for i in range(0, N, CHUNK):
            dist_matrix = torch.cdist(xyz1[i : i + CHUNK, :], xyz2)  # N(CHUNK),M
            _, top_k_nearest_idx[i : i + CHUNK, :] = torch.topk(dist_matrix, k=K, dim=1, largest=False)

    return top_k_nearest_idx.long()