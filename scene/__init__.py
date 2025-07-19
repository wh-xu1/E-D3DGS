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
import json
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.threshold_model import ThresholdModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.event_utils import load_event_stream, get_interpolator


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, dynamic_gaussians=None, static_load_path=None, dynamic_load_path=None, resolution_scales=[1.0]):

        self.args = args
        self.model_path = args.model_path
        self.static_load_path = static_load_path
        self.dynamic_load_path = dynamic_load_path
        self.gaussians = gaussians
        self.dynamic_gaussians = dynamic_gaussians

        self.train_cameras = {}
        self.test_cameras = {}

        
        if os.path.exists(os.path.join(args.source_path, "sparse")):                                # Colmap 
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms_train.json file, assuming Blender data set!")                   # Blender
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args) 
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if self.static_load_path is None:

            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):      
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras_center = -scene_info.nerf_normalization['translate']

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if args.event_assist:
            
            # Load Event stream
            self.event_list = load_event_stream(self.train_cameras[resolution_scale], args)

            # Get the interpolators
            self.rot_interpolator, self.trans_interpolator, self.tss_bd_start, self.tss_bd_end = get_interpolator(args)
                 
        if self.static_load_path is not None:
            self.gaussians.load_ply(
                    self.static_load_path, 
                    og_number_points=len(scene_info.point_cloud.points)
                )
            
            if self.dynamic_load_path is not None:
                self.dynamic_gaussians.load_ply(
                        self.dynamic_load_path, 
                        og_number_points=len(scene_info.point_cloud.points)
                    )
        
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        static_point_cloud_path = os.path.join(self.model_path, "static_point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(static_point_cloud_path, "point_cloud.ply"))

        if self.args.decomposition:
            dynamic_point_cloud_path = os.path.join(self.model_path, "dynamic_point_cloud/iteration_{}".format(iteration))
            self.dynamic_gaussians.save_ply(os.path.join(dynamic_point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getEvents(self):
        return self.event_list