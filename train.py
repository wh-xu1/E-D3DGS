import gc
import os

import torch
import numpy as np
from tqdm import tqdm
from random import randint
from torchvision import transforms
from collections import namedtuple
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from scene import DeformModel, ThresholdModel, GaussianModel
from scene.decomposition import *
from gaussian_renderer import render, render_from_merge

from utils import parse_utils
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.general_utils import safe_state
from utils.loss_utils import *
from utils.event_utils import *


def training(args):
    # TensorBoard
    tb_writer = prepare_logger(args)              

    # Init GS
    gaussians = GaussianModel(args.sh_degree)
    dynamic_gaussians = GaussianModel(args.sh_degree)   

    # Init scene
    scene = Scene(args, gaussians)
    gaussians.training_setup(args) 

    # Init DeformNet
    deform = DeformModel(args)
    deform.train_setting(args)

    # Init Threshold Model
    ThresholdNet = ThresholdModel(scene.getEvents().copy(), args)
    ThresholdNet.train_setting(args)

    # set background: black / white 
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Creating a Timer
    iter_start = torch.cuda.Event(enable_timing=True)   
    iter_end = torch.cuda.Event(enable_timing=True)

    train_start = torch.cuda.Event(enable_timing=True)
    train_end = torch.cuda.Event(enable_timing=True)

    # mics
    train_idx_stack = None
    ema_loss_for_log = 0.0
    ema_thres_loss_for_log = 0.0
    best_psnr = 0.0
    best_ssim = 0.0
    best_lpips = 1.0
    best_iteration = 0

    #######################################################################################
    ############################### Threshold Pre-train ###################################
    #######################################################################################

    pre_progress_bar = tqdm(range(args.thres_iteration), desc="Pre-training progress")

    # get training Camera
    train_camera_list = scene.getTrainCameras().copy()
    event_list = scene.getEvents().copy()
    total_frame = len(train_camera_list)

    for iter in range(1, args.thres_iteration +1):

        # sample training camera
        if not train_idx_stack:           
            train_idx_stack = list(range(total_frame)) 
        train_idx = train_idx_stack.pop(randint(0, len(train_idx_stack) - 1))

        ThresholdNet.optimizer.zero_grad()

        # sample event stream, compute thres loss
        thres_loss = ThresholdNet.pre_training(event_list, train_camera_list, train_idx, total_frame)
        tb_writer.add_scalar('pre-training/thres_loss', thres_loss.item(), iter)

        thres_loss.backward()
        ThresholdNet.optimizer.step()

        ema_thres_loss_for_log = 0.4 * thres_loss.item() + 0.6 * ema_thres_loss_for_log
        pre_progress_bar.set_postfix({"Thres_loss": f"{ema_thres_loss_for_log:.{7}f}"})
        pre_progress_bar.update(1)

    pre_progress_bar.close()

    #######################################################################################
    ################################### Start training ####################################
    #######################################################################################

    progress_bar = tqdm(range(args.iterations), desc="Training progress")

    # get training Camera
    train_camera_list = scene.getTrainCameras().copy()
    if args.event_assist:
        event_list = scene.getEvents().copy()
    total_frame = len(train_camera_list)    # 30
    
    train_start.record()
    for iteration in range(1, args.iterations + 1):
    
        #######################################################################################
        ####################################### DSD ###########################################
        #######################################################################################

        if iteration == args.warm_up: 

            # Select dynamic points
            sim_eval = SimilarityEvaluator(args).cuda().eval()
            init_dynamic_indices, prune_static_indices = decomposition(sim_eval, scene, gaussians, background, args)
                
            # Init dynamic GS
            point_cloud_path = os.path.join(args.model_path, 'dynamic_input_pcd.ply')
            gaussians.save_ply_with_indices(point_cloud_path, init_dynamic_indices)
            dynamic_gaussians.load_ply(point_cloud_path)
            scene.dynamic_gaussians = dynamic_gaussians
            dynamic_gaussians.training_setup(args) 

            # pure static GS
            prune_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool)
            prune_mask[prune_static_indices] = True 
            gaussians.prune_points(prune_mask)

            # Visualize dynamic GS and static GS
            if args.vis_decomposition:
                with torch.no_grad():
                    Path('vis_results/After_decomp/Only_static').mkdir(exist_ok=True, parents=True)
                    Path('vis_results/After_decomp/Only_dynamic').mkdir(exist_ok=True, parents=True)
                    remove_images('vis_results/After_decomp/Only_static')
                    remove_images('vis_results/After_decomp/Only_dynamic')

                    d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
                    for idx, train_camera in enumerate(train_camera_list):
                        render_pkg_re = render(train_camera, gaussians, args, background, d_xyz, d_rotation, d_scaling, args.is_6dof)
                        image = render_pkg_re["render"]
                        save_image = transforms.ToPILImage()(torch.clamp(image, 0.0, 1.0))
                        save_image.save(f'vis_results/After_decomp/Only_static/image_{idx:03d}.png')  

                        render_pkg_re = render(train_camera, dynamic_gaussians, args, background, d_xyz, d_rotation, d_scaling, args.is_6dof)
                        image = render_pkg_re["render"]
                        save_image = transforms.ToPILImage()(torch.clamp(image, 0.0, 1.0))
                        save_image.save(f'vis_results/After_decomp/Only_dynamic/image_{idx:03d}.png')      

            pass
        #######################################################################################

        iter_start.record()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()   

        # Sampling training cameras
        if not train_idx_stack:           
            train_idx_stack = list(range(total_frame))
        train_idx = train_idx_stack.pop(randint(0, len(train_idx_stack) - 1))
        viewpoint_cam = train_camera_list[train_idx]

        if args.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
    
        #######################################################################################
        ###################################### RGB Loss #######################################
        #######################################################################################

        fid = viewpoint_cam.fid * 1e-9

        if iteration < args.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            # render
            render_pkg_re = render(viewpoint_cam, gaussians, args, background, d_xyz, d_rotation, d_scaling, args.is_6dof)
        else:
            # deform
            N = dynamic_gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)     # M, 1
            d_xyz, d_rotation, d_scaling = deform.step(dynamic_gaussians.get_xyz.detach(), time_input)
            # joint training
            render_pkg_re = render_from_merge(viewpoint_cam, gaussians, dynamic_gaussians, args, background, d_xyz, d_rotation, d_scaling, args.is_6dof)
        
        image, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # compute RGB loss
        gt_image = viewpoint_cam.original_image.cuda()
        l1_loss = l1_loss_func(image, gt_image)
        rgb_loss = (1.0 - args.lambda_dssim) * l1_loss + args.lambda_dssim * (1.0 - ssim(image, gt_image))

        #######################################################################################
        ###################################### Event Loss #####################################
        #######################################################################################
 
        if (args.event_assist and iteration >= args.event_warm_up) and (iteration >= args.warm_up):
            
            # sample Events, compute thres loss
            thres_loss, ECM_sampled, threshold_sampled, event_tss, part_flag = ThresholdNet.joint_training(event_list, train_camera_list, train_idx, total_frame)

            # get event camera
            event_camera = interpolating_virtual_camera(float(event_tss), scene, viewpoint_cam, args)

            # Deform
            N = dynamic_gaussians.get_xyz.shape[0]
            event_time_input = (event_tss * 1e-9).cuda().unsqueeze(0).expand(N, -1).to(dtype=time_input.dtype)
            event_d_xyz, event_d_rotation, event_d_scaling = deform.step(dynamic_gaussians.get_xyz.detach(), event_time_input)
            # Joint render
            event_render_pkg_re = render_from_merge(event_camera, gaussians, dynamic_gaussians, args, background, event_d_xyz, event_d_rotation, event_d_scaling, args.is_6dof)

            image_at_event, visibility_filter_at_event, radii_at_event = event_render_pkg_re["render"], event_render_pkg_re["visibility_filter"], event_render_pkg_re["radii"]

            # Convert to grayscale
            gray_image = rgb_to_luma(viewpoint_cam.original_image.permute(1, 2, 0), esim=True)          # (H, W, 1)
            gray_image_at_event = rgb_to_luma(image_at_event.permute(1, 2, 0), esim=True)               # (H, W, 1)

            # Visualization
            if tb_writer and args.save_render_at_event and iteration % args.save_render_at_event_interval == 0:
                with torch.no_grad():
                    tb_writer.add_images("Event_loss/1_GT", gray_image.permute(2, 0, 1).unsqueeze(0), global_step=iteration)
                    tb_writer.add_images("Event_loss/2_Render_at_event", gray_image_at_event.permute(2, 0, 1).unsqueeze(0), global_step=iteration)

            # Inverse scale
            gray_image = gray_image * 255
            gray_image_at_event = gray_image_at_event * 255

            # Convert to Log space
            gray_image_log = lin_log(gray_image, linlog_thres=20)
            gray_image_at_event_log = lin_log(gray_image_at_event, linlog_thres=20)

            # Calculate image residual
            if part_flag == 'past':
                delta_linlog = (gray_image_log - gray_image_at_event_log)       # (H, W, 1)
            elif part_flag == 'future':
                delta_linlog = (gray_image_at_event_log - gray_image_log)       # (H, W, 1)

            event_frame = merge_ECM_with_thres(ECM_sampled, threshold_sampled)  # (H, W, 1)

            # compute Event loss
            image_event_residual = (delta_linlog - event_frame) ** 2
            event_loss = torch.mean(image_event_residual)

            # Visualization
            if tb_writer and args.save_render_at_event and iteration % args.save_render_at_event_interval == 0:
                with torch.no_grad():
                    # Event frame
                    min_val = torch.min(event_frame)
                    max_val = torch.max(event_frame)
                    norm_event_frame = ((event_frame - min_val) / (max_val - min_val))
                    tb_writer.add_images("Event_loss/4_Event_frame", norm_event_frame.permute(2, 0, 1).unsqueeze(0), global_step=iteration)

                    # Error Map of GT and Render Image
                    image_error_map = (delta_linlog ** 2).permute(2, 0, 1).unsqueeze(0)
                    tb_writer.add_images("Event_loss/3_Image_Image_Error_Map", image_error_map, global_step=iteration)

                    # Error Map for Image and Event
                    event_error_map = image_event_residual.permute(2, 0, 1).unsqueeze(0)
                    tb_writer.add_images("Event_loss/5_Image_Event_Error_Map", event_error_map, global_step=iteration)

            # total loss
            total_loss = rgb_loss + args.w_event * event_loss + args.w_thres * thres_loss

        # Only RGB Loss
        else:
            total_loss = rgb_loss
            event_loss = None
            thres_loss = None
            
        total_loss.backward()

        iter_end.record()   

        if args.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')


        #######################################################################################

        with torch.no_grad():

            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log 
            if iteration % 10 == 0:            
                progress_bar.set_postfix({"Total_loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)         
            if iteration == args.iterations:
                progress_bar.close()

            if (iteration < args.warm_up):
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                if event_loss is not None:
                    gaussians.max_radii2D[visibility_filter_at_event] = torch.max(gaussians.max_radii2D[visibility_filter_at_event], radii_at_event[visibility_filter_at_event])
            else:
                static_pcd_num = len(gaussians.get_xyz)
                gaussians.max_radii2D[visibility_filter[:static_pcd_num]] = torch.max(gaussians.max_radii2D[visibility_filter[:static_pcd_num]], radii[:static_pcd_num][visibility_filter[:static_pcd_num]])
                dynamic_gaussians.max_radii2D[visibility_filter[static_pcd_num:]] = torch.max(dynamic_gaussians.max_radii2D[visibility_filter[static_pcd_num:]], radii[static_pcd_num:][visibility_filter[static_pcd_num:]])

                if event_loss is not None:
                    gaussians.max_radii2D[visibility_filter_at_event[:static_pcd_num]] = torch.max(gaussians.max_radii2D[visibility_filter_at_event[:static_pcd_num]], radii_at_event[:static_pcd_num][visibility_filter_at_event[:static_pcd_num]])
                    dynamic_gaussians.max_radii2D[visibility_filter_at_event[static_pcd_num:]] = torch.max(dynamic_gaussians.max_radii2D[visibility_filter_at_event[static_pcd_num:]], radii_at_event[static_pcd_num:][visibility_filter_at_event[static_pcd_num:]])                


            #######################################################################################
            ######################################## Evaluation ###################################
            #######################################################################################
            
            psnr_test, ssim_test, lpips_test = training_report(
                    tb_writer, 
                    iteration,
                    rgb_loss,
                    event_loss,
                    thres_loss,
                    total_loss, 
                    iter_start.elapsed_time(iter_end),
                    args.testing_iterations, 
                    scene, 
                    render, 
                    render_from_merge,
                    (args, background), 
                    deform,
                    args.load2gpu_on_the_fly, 
                    args, 
                    args.is_6dof
                )
            
            # save best PSNR
            if iteration in args.testing_iterations: 
                if psnr_test.item() > best_psnr:
                    best_psnr = psnr_test.item() 
                    best_iteration = iteration
                
                    best_ssim = ssim_test.item()
                    best_lpips = lpips_test.item()

            # save scene
            if iteration in args.saving_iterations:  
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)


            #######################################################################################
            #################################### densify & pure ###################################
            #######################################################################################     
                       
            if iteration < args.densify_until_iter:

                if (iteration < args.warm_up):

                    # Accumulated Gradient
                    viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                    gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)
                    
                    if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:   
                        size_threshold = 20 if iteration > args.opacity_reset_interval else None
                        gaussians.densify_and_prune(args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)        

                    if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                        gaussians.reset_opacity()   
                else:
                    # Accumulated Gradient
                    viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                    gaussians.add_densification_stats_grad(viewspace_point_tensor_densify.grad[:static_pcd_num], visibility_filter[:static_pcd_num])
                    dynamic_gaussians.add_densification_stats_grad(viewspace_point_tensor_densify.grad[static_pcd_num:], visibility_filter[static_pcd_num:])

                    if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:   
                        size_threshold = 20 if iteration > args.opacity_reset_interval else None
                        gaussians.densify_and_prune(args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        dynamic_gaussians.densify_and_prune(args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)        

                    if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                        gaussians.reset_opacity()
                        dynamic_gaussians.reset_opacity()   


        # Optimizer Step
        gaussians.optimizer.step()
        gaussians.update_learning_rate(iteration)

        if (iteration >= args.warm_up):
            dynamic_gaussians.optimizer.step()
            dynamic_gaussians.update_learning_rate(iteration)                    

        deform.optimizer.step()
        deform.update_learning_rate(iteration)

        ThresholdNet.optimizer.step()
        ThresholdNet.update_learning_rate(iteration)


        # Optimizer Zero Grad
        gaussians.optimizer.zero_grad(set_to_none=True)

        if (iteration >= args.warm_up):
            dynamic_gaussians.optimizer.zero_grad(set_to_none=True)

        deform.optimizer.zero_grad(set_to_none=True)

        ThresholdNet.optimizer.zero_grad(set_to_none=True)


    # End of training
    train_end.record()
    gc.collect()
    torch.cuda.empty_cache()

    training_elapsed_min = train_start.elapsed_time(train_end) / 60000
    print("Total training time {} in Iteration {}".format(training_elapsed_min, best_iteration))
    tb_writer.add_scalar('total_training_time', training_elapsed_min, best_iteration)

    print("Best PSNR = {}, SSIM = {}, LPIPS = {} in Iteration {}".format(best_psnr, best_ssim, best_lpips, best_iteration))
    tb_writer.add_scalar('Best PSNR', best_psnr, best_iteration)
    tb_writer.add_scalar('Best SSIM', best_ssim, best_iteration)
    tb_writer.add_scalar('Best LPIPS', best_lpips, best_iteration)

    with open(os.path.join(args.model_path, f'PSNR_{best_psnr:.5f}_SSIM_{best_ssim:.5f}_LPIPS_{best_lpips:.5f}_time_{training_elapsed_min:.5f}_iter_{best_iteration:.5f}'), 'w', encoding='utf-8') as file:
        file.write("Best PSNR = {}, SSIM = {}, LPIPS = {} in Time {} Min and Iteration {}".format(best_psnr, best_ssim, best_lpips, training_elapsed_min, best_iteration))


def prepare_logger(args):
    
    # tb writer
    tb_writer = None
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer


def training_report(tb_writer, iteration, rgb_loss, event_loss, thres_loss, total_loss, elapsed, testing_iterations, scene: Scene, renderFunc, render_mergeFunc, renderArgs, deform, load2gpu_on_the_fly, args, is_6dof=False):

    # loss
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)                            

        tb_writer.add_scalar('train_loss_patches/rgb_loss', rgb_loss.item(), iteration)
        
        if event_loss is not None:
            tb_writer.add_scalar('train_loss_patches/event_loss', event_loss.item(), iteration)
            
        if thres_loss is not None:
            tb_writer.add_scalar('train_loss_patches/thres_loss', thres_loss.item(), iteration)

        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)   

    psnr_test, ssim_test, lpips_test = 0.0, 0.0, 1.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        images = torch.tensor([], device="cuda")
        gts = torch.tensor([], device="cuda")

        for idx, viewpoint in enumerate(scene.getTestCameras()):

            if load2gpu_on_the_fly:
                viewpoint.load2device()

            if iteration < args.warm_up:
                d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"], 0.0, 1.0)
            else:
                fid = viewpoint.fid * 1e-9
                xyz = scene.dynamic_gaussians.get_xyz
                time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                image = torch.clamp(render_mergeFunc(viewpoint, scene.gaussians, scene.dynamic_gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"], 0.0, 1.0)

            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            images = torch.cat((images, image.unsqueeze(0)), dim=0)         # 1, 3, H, W
            gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)            # 1, 3, H, W

            if load2gpu_on_the_fly:
                viewpoint.load2device('cpu')

            if tb_writer and (idx < 10): 
                tb_writer.add_images("test_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                if iteration == testing_iterations[0]:      
                    tb_writer.add_images("test_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                if (iteration >= args.warm_up):
                    static_image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, 0, 0, 0, is_6dof)["render"], 0.0, 1.0)
                    dynamic_image = torch.clamp(renderFunc(viewpoint, scene.dynamic_gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"], 0.0, 1.0)
                    tb_writer.add_images("test_view_{}_division/Static_image".format(viewpoint.image_name), static_image.unsqueeze(0), global_step=iteration)
                    tb_writer.add_images("test_view_{}_division/Dynamic_image".format(viewpoint.image_name), dynamic_image.unsqueeze(0), global_step=iteration)

        psnr_test = psnr(images, gts).mean()
        ssim_test = ssim(images, gts).mean()
        lpips_test = lpips(images, gts).mean() / images.shape[0]

        print("\n[ITER {}] Evaluating test: PSNR {}, SSIM {}, LPIPS {}".format(iteration, psnr_test, ssim_test, lpips_test))

        if tb_writer:
            tb_writer.add_scalar('test/loss_viewpoint - PSNR', psnr_test, iteration)
            tb_writer.add_scalar('test/loss_viewpoint - SSIM', ssim_test, iteration)
            tb_writer.add_scalar('test/loss_viewpoint - LPIPS', lpips_test, iteration)
        
        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            if (iteration >= args.warm_up):
                tb_writer.add_scalar('static_points_num', scene.gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar('dynamic_points_num', scene.dynamic_gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar('dynamic_points_ratio', scene.dynamic_gaussians.get_xyz.shape[0] / (scene.dynamic_gaussians.get_xyz.shape[0] + scene.gaussians.get_xyz.shape[0]), iteration)
                print('\n[ITER {}] Dynamic_points_num: {}'.format(iteration, scene.dynamic_gaussians.get_xyz.shape[0]))
                print('\n[ITER {}] Dynamic_points_ratio: {}'.format(iteration, scene.dynamic_gaussians.get_xyz.shape[0] / (scene.dynamic_gaussians.get_xyz.shape[0] + scene.gaussians.get_xyz.shape[0])))

        torch.cuda.empty_cache()

    return psnr_test, ssim_test, lpips_test


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--config', default='config/Debug.ini')
    parser.add_argument('--override', default=None)
    parser.add_argument("--local_rank", type=int)      

    args = parser.parse_args()

    # load ini
    cfg, print_format = parse_utils.parse_ini(args.config)

    # override
    if args.override is not None:
        cfg = parse_utils.override_cfg(args.override, cfg)

    # Processing individual items
    override_dict = {}

    override_dict['testing_iterations'] = list(range(1000, cfg.iterations + 1, 1000))
    override_dict['saving_iterations'] = list(range(5_000, cfg.iterations + 1, 5_000))

    # compute train_select & train_select
    image_root = Path(cfg.source_path) / 'rgb'
    train_select = np.linspace(0, len(list((image_root).rglob('*.*'))) - 1, cfg.train_rgb_num, dtype=int)
    test_select = np.zeros((len(train_select) - 1), dtype=np.int64)
    for i in range(cfg.train_rgb_num - 1):
        mid_point = round((train_select[i] + train_select[i + 1]) / 2)
        test_select[i] = mid_point        
        assert mid_point not in train_select, 'Test select Error!'

    override_dict['train_select'] = sorted(train_select.tolist())
    override_dict['test_select'] = sorted(test_select.tolist())
    
    cfg_dict = cfg._asdict()            
    Config = namedtuple('Config', tuple(set(cfg._fields + tuple(override_dict.keys()))))
    cfg_dict.update(override_dict) 
    cfg = Config(**cfg_dict)

    # print cfg
    parse_utils.print_cfg(cfg, print_format)

    # save cfg
    parse_utils.save_cfg(cfg, print_format)
    
    # Initialize system state (RNG)
    safe_state(cfg.quiet)

    torch.autograd.set_detect_anomaly(cfg.detect_anomaly) 

    training(cfg)
    
    print("\nTraining complete.")
