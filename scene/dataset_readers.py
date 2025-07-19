#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):

        cam_centers = np.hstack(cam_centers)    # 3, 200
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)        # 3, 1
        center = avg_cam_center

        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)  # L2归一化  1, 200
        diagonal = np.max(dist)     # 1
        return center.flatten(), diagonal   # center: 3     diagonal: 1

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)  # W2C: [R, T; 0, 1]
        C2W = np.linalg.inv(W2C)            
        cam_centers.append(C2W[:3, 3:4])   

    center, diagonal = get_center_and_diag(cam_centers)

    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):  
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,     
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)        

    vertices = plydata['vertex']        

    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T                  #  100_000, 3
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0    #  100_000, 3
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T               #  100_000, 3 
    normals = np.zeros((positions.shape[0], 3))

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):    
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,    
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, white_background, select, extension=".png"):

    cam_infos = []
    path = Path(path)

    with open(path, 'r') as json_file:
        contents = json.load(json_file)

        fovx = contents["camera_angle_x"]   

        frames = contents["frames"]

        for idx in select:
            frame = frames[idx]

            frame_idx = int(frame['file_path'].split('/')[-1])

            frame_time = frame['time'] * 1e9                                # 

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))     #  (4, 4)
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]                                              # R: (3, 3)
            T = -matrix[:3, 3]                                              # T: (3, )

            image_path = (path.parent / frame['file_path']).with_suffix(extension)
            image = Image.open(image_path)

            image_name = Path(image_path).name

            im_data = np.array(image.convert("RGBA"))                       # 400, 400, 4
            
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])     

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])  #  800, 800, 3
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy     # fovy==fovx

            cam_infos.append(
                CameraInfo(
                    uid=frame_idx,
                    R=R,
                    T=T,
                    FovY=FovY, 
                    FovX=FovX, 
                    image=image,
                    image_path=image_path, 
                    image_name=image_name, 
                    width=image.size[0],
                    height=image.size[1],
                    fid=frame_time
                    )
                )
            
    return cam_infos

 
def readNerfSyntheticInfo(path, white_background, eval, args, extension=".png"):


    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(os.path.join(path, 'transforms.json'), white_background, args.train_select, extension)

    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(os.path.join(path, 'transforms.json'), white_background, args.test_select, extension)

    if not eval:    
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)     

    if args.random_points:    
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        ply_path = os.path.join(args.model_path, 'input_pcd.ply')
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd_path = Path(args.pcd_path)

        if pcd_path.suffix == '.bin':
            xyz, rgb, _ = read_points3D_binary(pcd_path)
            ply_path = os.path.join(args.model_path, 'input_pcd.ply')
            storePly(ply_path, xyz, rgb)
        else:
            ply_path = pcd_path

    try:
        pcd = fetchPly(ply_path)        
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,                         
                           train_cameras=train_cam_infos,           # train_cameras
                           test_cameras=test_cam_infos,             # test_cameras
                           nerf_normalization=nerf_normalization,   
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
}
