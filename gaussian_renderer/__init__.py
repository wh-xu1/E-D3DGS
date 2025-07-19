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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None):

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0            # M, 3
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0    # M, 3

    try:
        screenspace_points.retain_grad()          
        screenspace_points_densify.retain_grad()    #
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),    # H 400
        image_width=int(viewpoint_camera.image_width),      # W 400
        tanfovx=tanfovx,                                  
        tanfovy=tanfovy,                                   
        bg=bg_color,                                       
        scale_modifier=scaling_modifier,                   
        viewmatrix=viewpoint_camera.world_view_transform,  
        projmatrix=viewpoint_camera.full_proj_transform,  
        sh_degree=pc.active_sh_degree,                    
        campos=viewpoint_camera.camera_center,            
        prefiltered=False,                                
        debug=pipe.debug,                             
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz + d_xyz
    
    opacity = pc.get_opacity            

    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc.get_scaling + d_scaling
    rotations = pc.get_rotation + d_rotation

    shs = None
    colors_precomp = None
    shs = pc.get_features              

    rendered_image, radii, depth = rasterizer(
        means3D=means3D,                           
        means2D=screenspace_points,              
        means2D_densify=screenspace_points_densify,
        shs=shs,                                   
        colors_precomp=colors_precomp,             
        opacities=opacity,                       
        scales=scales,                              
        rotations=rotations,                     
        cov3D_precomp=cov3D_precomp                
        )                

    return {"render": rendered_image,                                   
            "viewspace_points": screenspace_points,                    
            "viewspace_points_densify": screenspace_points_densify,    
            "visibility_filter": radii > 0,                          
            "radii": radii,                                             
            "depth": depth}                                            


def render_from_merge(viewpoint_camera, static_gaussians: GaussianModel, dynamic_gaussians: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None):


    screenspace_points = torch.zeros_like(torch.cat((static_gaussians.get_xyz, dynamic_gaussians.get_xyz), dim=0), dtype=static_gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0            # M, 3
    screenspace_points_densify = torch.zeros_like(screenspace_points, dtype=screenspace_points.dtype, requires_grad=True, device="cuda") + 0    # M, 3

    try:
        screenspace_points.retain_grad()           
        screenspace_points_densify.retain_grad()    
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),    
        image_width=int(viewpoint_camera.image_width),      
        tanfovx=tanfovx,                                   
        tanfovy=tanfovy,                                   
        bg=bg_color,                                      
        scale_modifier=scaling_modifier,                   
        viewmatrix=viewpoint_camera.world_view_transform,  
        projmatrix=viewpoint_camera.full_proj_transform,   
        sh_degree=static_gaussians.active_sh_degree,       
        campos=viewpoint_camera.camera_center,            
        prefiltered=False,                                 
        debug=pipe.debug,                                  
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = torch.cat((static_gaussians.get_xyz, dynamic_gaussians.get_xyz + d_xyz), dim=0)
    
    opacity = torch.cat((static_gaussians.get_opacity, dynamic_gaussians.get_opacity), dim=0)            


    scales = None
    rotations = None
    cov3D_precomp = None
    scales = torch.cat((static_gaussians.get_scaling, dynamic_gaussians.get_scaling + d_scaling), dim=0)
    rotations = torch.cat((static_gaussians.get_rotation, dynamic_gaussians.get_rotation + d_rotation), dim=0)

    shs = None
    colors_precomp = None
    shs = torch.cat((static_gaussians.get_features, dynamic_gaussians.get_features), dim=0) 


    rendered_image, radii, depth = rasterizer(     
        means3D=means3D,                           
        means2D=screenspace_points,              
        means2D_densify=screenspace_points_densify,
        shs=shs,                                  
        colors_precomp=colors_precomp,              
        opacities=opacity,                          
        scales=scales,                             
        rotations=rotations,                      
        cov3D_precomp=cov3D_precomp             
        )                

    return {"render": rendered_image,                                 
            "viewspace_points": screenspace_points,                    
            "viewspace_points_densify": screenspace_points_densify,    
            "visibility_filter": radii > 0,                            
            "radii": radii,                                           
            "depth": depth}                                           
