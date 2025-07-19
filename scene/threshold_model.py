import torch
import random
import numpy as np
import torch.nn as nn
from utils.event_utils import rgb_to_luma, merge_ECM_with_thres, lin_log
from utils.general_utils import get_expon_lr_func


class ThresholdModel:
    def __init__(self, event_list, args):
        self.thresholdnet = ThresholdNet(event_list, args).cuda()
    
    def pre_training(self, event_list, train_camera_list, train_idx, train_camera_num):
        return self.thresholdnet.pre_training(event_list, train_camera_list, train_idx, train_camera_num)

    def joint_training(self, event_list, train_camera_list, train_idx, train_camera_num):
        return self.thresholdnet.joint_training(event_list, train_camera_list, train_idx, train_camera_num)

    def train_setting(self, args): 
        l = [{
            'params': list(self.thresholdnet.parameters()),
            'lr': args.thresholdnet_pretrain_lr,
            "name": "thresholdnet"
            }]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.thresholdnet_scheduler = get_expon_lr_func(
                                            lr_init=args.thresholdnet_ft_lr_init,
                                            lr_final=args.thresholdnet_ft_lr_final,
                                            max_steps=args.thresholdnet_ft_lr_max_steps,
                                        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "thresholdnet":
                lr = self.thresholdnet_scheduler(iteration)
                param_group['lr'] = lr
                return lr


class ThresholdNet(nn.Module):
    def __init__(self, event_list, args):
        super(ThresholdNet, self).__init__()
        self.args = args
        
        threshold_list = []
        for events in event_list:
            ECM = events['ECM']                 # 2, B, H, W

            threshold = nn.Parameter(torch.ones_like(ECM) * args.threshold_init)    # 2, B, H, W
            threshold_list.append(threshold)
        
        self.threshold_list = nn.ParameterList(threshold_list)

    def past_or_future(self, event_list, train_camera_list, train_idx, train_camera_num):
        if (train_idx != 0 and np.random.random() < 0.5) or (train_idx == train_camera_num - 1):

            ECM = event_list[train_idx-1]['ECM'].cuda()             # 2, B, H, W
            threshold = self.threshold_list[train_idx-1]            # 2, B, H, W
            tss_list = event_list[train_idx-1]['tss']               # B+1

            last_image = train_camera_list[train_idx-1].original_image.cuda()   # 3, H, W
            next_image = train_camera_list[train_idx].original_image.cuda()     # 3, H, W

            part_flag = 'past'

        else:
            ECM = event_list[train_idx]['ECM'].cuda()
            threshold = self.threshold_list[train_idx]
            tss_list = event_list[train_idx]['tss']

            last_image = train_camera_list[train_idx].original_image.cuda()
            next_image = train_camera_list[train_idx+1].original_image.cuda()

            part_flag = 'future'

        return ECM, threshold, tss_list, part_flag, last_image, next_image

    def sampling_event(self, ECM, tss_list, threshold, part_flag):
        """
            Sample sub event stream
        """

        # Randomly sample an event stream from the past or future of the RGB frame
        if part_flag == 'past':
            sampled_idx = random.choice(range(1, self.args.bin_num))

            ECM_sampled = ECM[:, -sampled_idx:, :, :]                   # 2, M, H, W
            threshold_sampled = threshold[:, -sampled_idx:, :, :]       # 2, M, H, W
            tss_sampled = tss_list[-(sampled_idx + 1)]

        elif part_flag == 'future':
            sampled_idx = random.choice(range(1, self.args.bin_num))

            ECM_sampled = ECM[:, :sampled_idx, :, :]                    # 2, M, H, W
            threshold_sampled = threshold[:, :sampled_idx, :, :]        # 2, M, H, W
            tss_sampled = tss_list[sampled_idx]

        return ECM_sampled, threshold_sampled, tss_sampled

    def get_thres_loss(self, ECM, threshold, last_image, next_image):
        
        # Go to L Space
        last_image_gray = rgb_to_luma(last_image.permute(1, 2, 0))  # H, W, 1
        next_image_gray = rgb_to_luma(next_image.permute(1, 2, 0))  # H, W, 1
        
        # Inverse scale
        last_image_gray = last_image_gray * 255
        next_image_gray = next_image_gray * 255
    
        # Convert to Log space
        last_image_gray_log = lin_log(last_image_gray, linlog_thres=20)
        next_image_gray_log = lin_log(next_image_gray, linlog_thres=20)
        
        error_map = (next_image_gray_log - last_image_gray_log)     # H, W, 1

        event_frame = merge_ECM_with_thres(ECM, threshold)          # H, W, 1

        image_event_residual = (error_map - event_frame) ** 2

        thres_loss = torch.mean(image_event_residual)

        return thres_loss
    
    def pre_training(self, event_list, train_camera_list, train_idx, train_camera_num):
        # sample event stream
        ECM, threshold, tss_list, part_flag, last_image, next_image = self.past_or_future(event_list, train_camera_list, train_idx, train_camera_num)
        # compute thres loss
        thres_loss = self.get_thres_loss(ECM, threshold, last_image, next_image)

        return thres_loss
    
    def joint_training(self, event_list, train_camera_list, train_idx, train_camera_num):
        # sample event stream
        ECM, threshold, tss_list, part_flag, last_image, next_image = self.past_or_future(event_list, train_camera_list, train_idx, train_camera_num)
        # sample sub event stream
        ECM_sampled, threshold_sampled, tss_sampled = self.sampling_event(ECM, tss_list, threshold, part_flag)
        # compute thres loss
        thres_loss = self.get_thres_loss(ECM, threshold, last_image, next_image)

        return thres_loss, ECM_sampled, threshold_sampled, tss_sampled, part_flag