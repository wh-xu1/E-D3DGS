from pathlib import Path
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import os
import torch
import json
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d
from scene.cameras import Camera


def binary_search_torch_tensor(t, l, r, x, side='left'):

    if r is None:
        r = len(t)-1
    while l <= r:
        if t[l] == x:
            return l
        if t[r] == x:
            return r
            
        mid = l + (r - l)//2
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

def events_to_image(xs, ys, ps, sensor_size=(180, 240)):

    xs_mask = (xs >= sensor_size[1]) + (xs < 0) 
    ys_mask = (ys >= sensor_size[0]) + (ys < 0) 
    mask = xs_mask + ys_mask
    xs[mask] = 0
    ys[mask] = 0
    ps[mask] = 0

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device, dtype=torch.int64)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)

    return img

def events_to_stack(xs, ys, ts, ps, B, sensor_size=(400, 400)):

    tss_list = []
    if ts.sum() == 0 or len(ts) <= 3:
        return torch.zeros([2, B, sensor_size[0], sensor_size[1]])

    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    positives = []
    negtives = []
    dt = ts[-1] - ts[0]
    delta_t = dt / B
    for bi in range(B):
        tstart = ts[0] + delta_t * bi
        tss_list.append(torch.ceil(tstart).to(dtype=torch.int64))
        tend = tstart + delta_t
        beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart) 
        end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend, side='right') + 1

        mask_pos = ps[beg:end].clone()
        mask_neg = ps[beg:end].clone()
        mask_pos[ps[beg:end] < 0] = 0
        mask_neg[ps[beg:end] > 0] = 0

        vp = events_to_image(xs[beg:end], ys[beg:end], ps[beg:end] * mask_pos, sensor_size=sensor_size)
        vn = events_to_image(xs[beg:end], ys[beg:end], ps[beg:end] * mask_neg, sensor_size=sensor_size)

        positives.append(vp)
        negtives.append(vn)

    tss_list.append(torch.floor(tend).to(dtype=torch.int64))
    
    positives_b = torch.stack(positives)
    negtives_b = torch.stack(negtives)
    stack = torch.stack([positives_b, negtives_b])
    tss_list = torch.stack(tss_list)

    return stack, tss_list

def add_noise(ECM, noise_std=1.0, noise_fraction=0.05):
    # ECM: 2, B, H, W
    
    noise = (noise_std * torch.randn_like(ECM.float())).abs().int()  # mean = 0, std = noise_std
    if noise_fraction < 1.0:
        mask = torch.rand_like(ECM.float()) >= noise_fraction
        noise.masked_fill_(mask, 0)

    return ECM + noise

def load_event_stream(sorted_cameras, args):

    frame_id_list = [cam.colmap_id for cam in sorted_cameras]
    
    event_root = Path(args.source_path) / 'events'

    j = 1
    event_acc = None
    event_list = []
    
    print('Loading Training Event stream')
    for i, event_path in tqdm(enumerate(sorted(event_root.iterdir()))):

        event_i = np.load(event_path)               
        event_i = np.vstack([event_i['x'], event_i['y'], event_i['t'], event_i['p']]).T
        
        if event_acc is None:
            event_acc = event_i
        else:
            event_acc = np.concatenate((event_acc, event_i), axis=0)  

        if j < len(frame_id_list):            
            if i == frame_id_list[j] - 1:            
                
                event_acc = torch.from_numpy(event_acc)
                ECM, tss_ns = events_to_stack(xs=event_acc[:, 0], ys=event_acc[:, 1], ts=event_acc[:, 2], ps=event_acc[:, 3], B=args.bin_num, sensor_size=(args.event_height, args.event_width))
                
                if args.spatially_varying_thresholds:
                    ECM = add_noise(ECM, noise_fraction=args.noise_fraction)

                event_list.append({'ECM': ECM, 'tss': tss_ns})

                event_acc = None
                j += 1

    return event_list 

def rgb_to_luma(rgb, esim=True):

    device = rgb.device

    if esim:
        #  https://github.com/uzh-rpg/rpg_esim/blob/4cf0b8952e9f58f674c3098f1b027a4b6db53427/event_camera_simulator/imp/imp_opengl_renderer/src/opengl_renderer.cpp#L319-L321
        #  image format esim: https://github.com/uzh-rpg/rpg_esim/blob/4cf0b8952e9f58f674c3098f1b027a4b6db53427/event_camera_simulator/esim_visualization/src/ros_utils.cpp#L29-L36
        #  color conv factorsr rgb->gray: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
        r = 0.299
        g = 0.587
        b = 0.114
    else:
        r = 0.2126
        g = 0.7152
        b = 0.0722

    factors = torch.Tensor([r, g, b]).to(device)  # (3)
    luma = torch.sum(rgb * factors[None, :], axis=-1)  # (N_evs, 3) * (1, 3) => (N_evs)
    return luma[..., None]  # (N_evs, 1)

def lin_log(color, linlog_thres=20):
    """
    Input: 
    :color torch.Tensor of (N_rand_events, 1 or 3). 1 if use_luma, else 3 (rgb).
           We pass rgb here, if we want to treat r,g,b separately in the loss (each pixel must obey event constraint).
    """
    # Compute the required slope for linear region (below luma_thres)
    # we need natural log (v2e writes ln and "it comes from exponential relation")
    lin_slope = np.log(linlog_thres) / linlog_thres

    # Peform linear-map for smaller thres, and log-mapping for above thresh
    lin_log_rgb = torch.where(color < linlog_thres, lin_slope * color, torch.log(color))
    return lin_log_rgb

def make_event_frame(event_tensor, resolution):
    # Accumulate events to create a 1 * H * W image

    H, W = resolution

    assert (event_tensor != 0).all()
    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)

    event_count = pos_count - neg_count

    result = torch.unsqueeze(event_count, -1)
    result = result.float()

    return result

def get_interpolator(args):

    transforms_path = os.path.join(args.source_path, 'transforms.json')

    with open(transforms_path, 'r') as json_file:

        contents = json.load(json_file)
        frames = contents["frames"]

        rots = []
        trans = []
        tss_ns = []

        for frame in frames:
            tss_ns.append(np.array([frame['time'] * 1e9]))

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            
            Rt = np.zeros((4, 4))                           # W2C: 4, 4
            Rt[:3, :3] = R.transpose()                      
            Rt[:3, 3] = T
            Rt[3, 3] = 1.0

            C2W = np.linalg.inv(Rt)                         # C2W: 4, 4

            rots.append(C2W[:3, :3])                        # [(3, 3), ...]
            trans.append(C2W[:3, 3])                        # [(3, ), ...]

        rots = np.stack(rots)                   # 4000, 3, 3
        trans = np.stack(trans)                 # 4000, 3
        tss_ns = np.stack(tss_ns).squeeze()     # 4000,

        rot_interpolator = Slerp(tss_ns, Rotation.from_matrix(rots))
        trans_interpolator = interp1d(x=tss_ns, y=trans, axis=0, kind="cubic", bounds_error=False)

    tss_bd_start = np.min(rot_interpolator.times)
    tss_bd_end = np.max(rot_interpolator.times)
        
    return rot_interpolator, trans_interpolator, tss_bd_start, tss_bd_end

def RT_from_rots_trans(rots, trans):
    assert rots.shape == (3, 3)   
    assert trans.shape == (3, )
    
    rt_C2W = np.zeros((4, 4))
    hom = np.array([0,0,0,1]).reshape((1, 4))

    rt_C2W[:3, :3] = rots.copy()     # (3, 3)
    rt_C2W[:3, 3] = trans.copy()     # (3, 1)
    rt_C2W[3, :] = hom.copy()        # (1, 4)

    rt_C2W = torch.from_numpy(rt_C2W)   # (4, 4)

    RT_W2C = np.linalg.inv(rt_C2W)      # (4, 4)
    R = RT_W2C[:3, :3].transpose()      # (3, 3)    
    T = RT_W2C[:3, 3]                   # (3, )

    return R, T

def sampling_event(event_list, train_idx, train_camera_num, args):

    if (train_idx != 0 and np.random.random() < 0.5) or (train_idx == train_camera_num - 1):

        ECM = event_list[train_idx-1]['ECM'].cuda()               # 2, B, H, W
        tss_list = event_list[train_idx-1]['tss']                 # B+1
        sampled_idx = random.choice(range(1, args.bin_num))

        ECM_sampled = ECM[:, -sampled_idx:, :, :]               # 2, M, H, W
        sampled_tss = tss_list[-(sampled_idx + 1)]

        part_flag = 'past'

    else:
        ECM = event_list[train_idx]['ECM'].cuda()
        tss_list = event_list[train_idx]['tss']
        sampled_idx = random.choice(range(1, args.bin_num))

        ECM_sampled = ECM[:, :sampled_idx, :, :]
        sampled_tss = tss_list[sampled_idx]

        part_flag = 'future'
    
    return ECM_sampled, sampled_tss, part_flag

def interpolating_virtual_camera(event_tss, scene, viewpoint_cam, args):
    if event_tss < scene.tss_bd_start:
        print(f'The timestamp {event_tss} of the sampled event is earlier than the interpolator\'s start time {scene.tss_bd_start}!')
        event_tss = scene.tss_bd_start

    elif event_tss > scene.tss_bd_end:
        print(f'The timestamp {event_tss} of the sampled event is later than the interpolator\'s end time {scene.tss_bd_end}!')

        event_tss = scene.tss_bd_end
    
    rots_at_event = scene.rot_interpolator(event_tss).as_matrix()  # (3, 3)
    trans_at_event = scene.trans_interpolator(event_tss)           # (3, )

    R_at_event, T_at_event = RT_from_rots_trans(rots_at_event, trans_at_event)

    event_camera = Camera(
            colmap_id=None,             
            R=R_at_event,
            T=T_at_event,                    
            FoVx=viewpoint_cam.FoVx,
            FoVy=viewpoint_cam.FoVy,
            image=torch.zeros(3, args.event_height, args.event_width),
            gt_alpha_mask=None,
            image_name=None, 
            uid=None,  
            data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu',  # 'cuda'
            fid=event_tss,      
            depth=None       
    )

    return event_camera

def merge_ECM_with_thres(ECM, threshold):

    ECM_with_thres = ECM * threshold    # 2, B, H, W

    pos = ECM_with_thres[0, :, :, :]    # M, H, W
    neg = ECM_with_thres[1, :, :, :]    # M, H, W

    pos = torch.sum(pos, dim=0)         # H, W
    neg = torch.sum(neg, dim=0)         # H, W

    merge_event_frame = pos - neg
    merge_event_frame = merge_event_frame.unsqueeze(-1)     # H, W, 1

    return merge_event_frame
    
def find_max_connected_components(array):
    G = nx.Graph()
    G.add_edges_from(array)
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    components = [list(component) for component in connected_components]

    return components