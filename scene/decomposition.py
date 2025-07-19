import os
import cv2
import glob
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
from plyfile import PlyData, PlyElement

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.project_utils import unproject
from utils.event_utils import find_max_connected_components


def remove_images(path):
    image_files = glob.glob(os.path.join(path, "*.png"))
    for image_file in image_files:
        os.remove(image_file)

class SimilarityEvaluator(nn.Module):
    def __init__(self, args):
        super(SimilarityEvaluator, self).__init__()

        vgg19 = models.vgg19(pretrained=False)
        state_dict = torch.load(args.vgg19_path)
        vgg19.load_state_dict(state_dict)

        self.selected_layers = [8,17,26]
        self.model = nn.Sequential(*list(vgg19.features.children())[:max([8,17,26]) + 1])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, x):
        features = []
        for name, layer in self.model._modules.items():
            x = layer(x)
            if int(name) in self.selected_layers:
                features.append(x)
        return features
    
    def forward(self, syn: torch.Tensor, gt: torch.Tensor):
        
        syn = self.transform(syn).unsqueeze(0)  # torch.Size([1, 3, 224, 224])
        gt = self.transform(gt).unsqueeze(0)    # torch.Size([1, 3, 224, 224])

        with torch.no_grad():
            syn_features = self.extract_features(syn)   # [torch.Size([1, 128, 112, 112]), torch.Size([1, 256, 56, 56]), torch.Size([1, 512, 28, 28])]
            gt_features = self.extract_features(gt)
        
        similarity_maps = []
        for syn_feat, gt_feat in zip(syn_features, gt_features):
            sim_map = torch.nn.functional.cosine_similarity(syn_feat, gt_feat, dim=1)   # torch.Size([1, 112, 112])
            upsampled_map = nn.functional.interpolate(sim_map.unsqueeze(1), size=(400, 400), mode='bilinear', align_corners=False)  # torch.Size([1, 1, 224, 224])
            upsampled_map = upsampled_map.squeeze(1)    # torch.Size([1, 224, 224])
            similarity_maps.append(upsampled_map)
        
        similarity_maps = torch.stack(similarity_maps)                      # torch.Size([3, 1, 224, 224])
        similarity_maps = torch.mean(similarity_maps, dim=0)                # torch.Size([1, 224, 224])

        return similarity_maps


def decomposition(sim_eval: SimilarityEvaluator, scene: Scene, gaussians: GaussianModel, background: torch.Tensor, args):

    if args.vis_decomposition:
        Path('vis_results/Before_decomp/Render_all').mkdir(exist_ok=True, parents=True)
        Path('vis_results/Before_decomp/GT').mkdir(exist_ok=True, parents=True)
        Path('vis_results/After_decomp/Masked_dynamic').mkdir(exist_ok=True, parents=True)
        Path('vis_results/After_decomp/Mask').mkdir(exist_ok=True, parents=True)
        Path('vis_results/After_decomp/depth').mkdir(exist_ok=True, parents=True)
        Path('vis_results/Before_decomp/Similarity').mkdir(exist_ok=True, parents=True)

        remove_images('vis_results/Before_decomp/Render_all')
        remove_images('vis_results/Before_decomp/GT')
        remove_images('vis_results/After_decomp/Masked_dynamic')
        remove_images('vis_results/After_decomp/Mask')
        remove_images('vis_results/After_decomp/depth')
        remove_images('vis_results/Before_decomp/Similarity')

    with torch.no_grad():
        train_camera_list = scene.getTrainCameras().copy()

        unproject_pts_list = []
        for idx, train_camera in enumerate(train_camera_list):
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            render_pkg_re = render(train_camera, gaussians, args, background, d_xyz, d_rotation, d_scaling, args.is_6dof)
            image = render_pkg_re["render"]
            depth = render_pkg_re["depth"]
            GT = train_camera.original_image                    # torch.Size([3, 400, 400])

            if args.vis_decomposition:
                save_image = transforms.ToPILImage()(torch.clamp(image, 0.0, 1.0))
                save_image.save(f'vis_results/Before_decomp/Render_all/image_{idx:03d}.png')

                save_image = transforms.ToPILImage()(torch.clamp(GT, 0.0, 1.0))
                save_image.save(f'vis_results/Before_decomp/GT/image_{idx:03d}.png')

            similarity_map = sim_eval(image, GT)                # torch.Size([1, 400, 400])

            if args.vis_decomposition:
                save_image = transforms.ToPILImage()(torch.clamp(similarity_map, 0.0, 1.0))
                save_image.save(f'vis_results/Before_decomp/Similarity/image_{idx:03d}.png')

            ret, _ = cv2.threshold((similarity_map.squeeze().cpu().numpy()*255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
            ret = ret / 255.0
            dynamic_mask = (similarity_map < ret).float()       # torch.Size([1, 400, 400])

            white_bg = torch.ones_like(image) if args.white_background else torch.zeros_like(image)
            dynamic_image = dynamic_mask * image + (1 - dynamic_mask) * white_bg

            if args.vis_decomposition:
                save_image = transforms.ToPILImage()(torch.clamp(dynamic_image, 0.0, 1.0))
                save_image.save(f'vis_results/After_decomp/Masked_dynamic/image_{idx:03d}.png')

                save_image = transforms.ToPILImage()(torch.clamp(dynamic_mask, 0.0, 1.0))
                save_image.save(f'vis_results/After_decomp/Mask/image_{idx:03d}.png')
            
            pts = unproject(depth, train_camera.w2c, train_camera.k)    # torch.Size([3, 400, 400])
            pts = pts.reshape(pts.shape[0], -1)                         # torch.Size([3, 160000])
            unproject_pts = pts[:, dynamic_mask.reshape(-1).bool()].T     # torch.Size([10907, 3])
            unproject_pts_list.append(unproject_pts)

            if args.vis_decomposition:
                depth = depth / depth.max()
                black_bg = torch.zeros_like(depth)
                masked_depth = dynamic_mask * depth + (1 - dynamic_mask) * black_bg
                save_image = transforms.ToPILImage()(torch.clamp(masked_depth, 0.0, 1.0))
                save_image.save(f'vis_results/After_decomp/depth/image_{idx:03d}.png')

        all_unproject_pts = torch.cat(unproject_pts_list)

        distances = torch.norm(all_unproject_pts - torch.from_numpy(scene.cameras_center).cuda(), dim=1)
        all_unproject_pts = all_unproject_pts[distances <= 0.6 * scene.cameras_extent]

        all_unproject_pts_np = all_unproject_pts.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(all_unproject_pts_np)
        distances, _ = nbrs.kneighbors(all_unproject_pts_np)
        nearest_distances = distances[:, 1]

        high_bds = nearest_distances.mean() - args.outlier_filtering_strength * nearest_distances.std()
        unproject_pts = all_unproject_pts_np[nearest_distances < high_bds]

        if args.vis_decomposition:
            vertex = np.array([tuple(point) for point in unproject_pts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            ply_data = PlyData([PlyElement.describe(vertex, 'vertex')])
            ply_data.write('vis_results/unproject_pcd.ply')

        gs_pts_np = gaussians.get_xyz.detach().cpu().numpy()
        gs_nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(gs_pts_np)
        gs_dist, _ = gs_nbrs.kneighbors(gs_pts_np)
        manifold_radius = gs_dist[:, 1].mean()

        tree_unproject = cKDTree(unproject_pts)
        distances, _ = tree_unproject.query(gs_pts_np, k=1)
        dynamic_mask = distances < args.dynamic_radius_scale * manifold_radius
        buffer_mask = distances < args.buffer_radius_scale * manifold_radius

        if args.vis_decomposition:
            filtered_gs_pts = gs_pts_np[dynamic_mask]
            vertex = np.array([tuple(point) for point in filtered_gs_pts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            ply_data = PlyData([PlyElement.describe(vertex, 'vertex')])
            ply_data.write('vis_results/dynamic_pcd.ply')

        return dynamic_mask, buffer_mask