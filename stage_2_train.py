from __future__ import print_function, division
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm


from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torchvision.utils as vutils


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from AsymKD.dpt_latent1_avg_ver import AsymKD_compress_latent1_avg_ver
from core.loss import GradL1Loss, ScaleAndShiftInvariantLoss
from torch.optim.lr_scheduler import LambdaLR
from diffusion import GaussianDiffusion, load_diffusion_model
from einops import rearrange

import core.AsymKD_datasets as datasets
import gc

from dataset.util.alignment_gpu import align_depth_least_square
import torch.nn.functional as F
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def compute_errors(flow_gt, flow_preds, valid_arr):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    a1_arr = []
    a2_arr = []
    a3_arr = []
    abs_rel_arr = []
    rmse_arr = []
    # log_10_arr = []
    # rmse_log_arr = []
    # silog_arr = []
    sq_rel_arr = []

    min_depth_eval = 0.0001
    max_depth_eval = 1

    for gt, pred, valid in zip(flow_gt, flow_preds, valid_arr):
        
        disparity_pred, scale, shift = align_depth_least_square(
            gt_arr=gt,
            pred_arr=pred,
            valid_mask_arr=valid,
            return_scale_shift=True,
        )
        gt = gt.squeeze().cpu().numpy()
        pred = disparity_pred.clone().squeeze().cpu().detach().numpy()
        valid = valid.squeeze().cpu()        
        pred[pred < min_depth_eval] = min_depth_eval
        pred[pred > max_depth_eval] = max_depth_eval
        pred[np.isinf(pred)] = max_depth_eval
        pred[np.isnan(pred)] = min_depth_eval

        # pred[pred < min_depth_eval] = min_depth_eval
        # pred[pred > max_depth_eval] = max_depth_eval
        # pred[np.isinf(pred)] = max_depth_eval
        # pred[np.isnan(pred)] = min_depth_eval
        # gt[gt < min_depth_eval] = min_depth_eval
        # gt[gt > max_depth_eval] = max_depth_eval
        # gt[np.isinf(gt)] = max_depth_eval
        # gt[np.isnan(gt)] = min_depth_eval

        gt, pred= gt[valid.bool()], pred[valid.bool()]

        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = (np.abs(gt - pred) / gt).mean()
        sq_rel =(((gt - pred) ** 2) / gt).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        # rmse_log = (np.log(gt) - np.log(pred)) ** 2
        # rmse_log = np.sqrt(rmse_log.mean())

        # err = np.log(pred) - np.log(gt)
        # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        # log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

        a1_arr.append(a1)
        a2_arr.append(a2)
        a3_arr.append(a3)
        abs_rel_arr.append(abs_rel)
        rmse_arr.append(rmse)
        # log_10_arr.append(log_10)
        # rmse_log_arr.append(rmse_log)
        # silog_arr.append(silog)
        sq_rel_arr.append(sq_rel)

    a1_arr_mean = sum(a1_arr) / len(a1_arr)
    a2_arr_mean = sum(a2_arr) / len(a2_arr)
    a3_arr_mean = sum(a3_arr) / len(a3_arr)
    abs_rel_arr_mean = sum(abs_rel_arr) / len(abs_rel_arr)
    rmse_arr_mean = sum(rmse_arr) / len(rmse_arr)
    # log_10_arr_mean = sum(log_10_arr) / len(log_10_arr)
    # rmse_log_arr_mean = sum(rmse_log_arr) / len(rmse_log_arr)
    # silog_arr_mean = sum(silog_arr) / len(silog_arr)
    sq_rel_arr_mean = sum(sq_rel_arr) / len(sq_rel_arr)

    # return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, log_10=log_10_arr_mean, rmse_log=rmse_log_arr_mean,
    #             silog=silog_arr_mean, sq_rel=sq_rel_arr_mean)
    return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, sq_rel=sq_rel_arr_mean)

def sequence_loss(flow_preds, flow_gt, valid, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    # L1 loss
    flow_loss = F.l1_loss(flow_preds[valid.bool()], flow_gt[valid.bool()])
    metrics = compute_errors(flow_gt, flow_preds, valid)

    
    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #         pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=8e-4, 
        weight_decay=0.02,
        betas=(0.9, 0.95)
    )

    def lr_schedule(steps):
        if steps < (warmup_steps:=args.num_steps * 0.1):
            # Linear warmup
            return steps / warmup_steps
        else:
            # Constant learning rate
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

class State(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def capture(self):
        return {
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
    
    def apply_snapshot(self, obj, device):
        self.model.load_state_dict(obj['model_state_dict'].to(device), strict=False)
        self.optimizer.load_state_dict(obj['optimizer_state_dict'].to(device))
        self.scheduler.load_state_dict(obj['scheduler_state_dict'].to(device))
    
    def save(self, path):
        torch.save(self.capture(), path)
    
    def load(self, path, device):
        obj = torch.load(path, map_location=torch.device('cpu'))
        self.apply_snapshot(obj, device)

def train(rank, world_size, args):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

        # init scalars
        save_step = 2500
        total_steps = 0
        epoch = 0

        # load model - AsymKD_Compress
        AsymKD_Compress = AsymKD_compress_latent1_avg_ver().to(rank)
        # student_ckpt = 'depth_anything_vits14.pth'
        # teacher_ckpt = 'depth_anything_vitl14.pth'
        # if rank == 0:
        #     logging.info(f"loading backbones from {student_ckpt} and {teacher_ckpt}")
        # AsymKD_Compress.load_backbone_from_ckpt(student_ckpt, teacher_ckpt, device=torch.device('cuda', rank))
        # AsymKD_Compress = torch.nn.SyncBatchNorm.convert_sync_batchnorm(AsymKD_Compress)
        best_ckpts = torch.load('best_checkpoint_depth_latent1/depth_latent1_avg_best_checkpoint.pth', map_location=torch.device('cuda', rank))
        ckpt = {k.replace('module.', ''): v for k, v in best_ckpts['model_state_dict'].items()}
        AsymKD_Compress.load_state_dict(ckpt)
        AsymKD_Compress = AsymKD_Compress.eval()

        # load model - Diffusion
        model = DiffusionMLP(
            in_channels=384,
            out_channels=384,
            mid_channels=1024,
            num_resblks=6
        ).to(rank)
        model = GaussianDiffusion(
            model,
            seq_length=37*37,
            sampling_timesteps=10,
            objective="pred_v",
        ).to(rank)

        #model.module.freeze_bn() # We keep BatchNorm frozen
        model = DDP(model, device_ids=[rank])
        model.train()
        
        if rank == 0:
            logging.info(f"Parameter Count: {count_parameters(model)}")
            logging.info("AsymKD_VIT Train")
        
        # load others
        train_loader = datasets.fetch_dataloader(args,rank, world_size)
        optimizer, scheduler = fetch_optimizer(args, model)
        scaler = GradScaler(enabled=args.mixed_precision)
        logger = Logger(model, scheduler)
        state = State(model, optimizer, scheduler)

        # load loss
        # SSILoss = ScaleAndShiftInvariantLoss()
        # grad_loss = GradL1Loss()

        # load snapshot
        if args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth")
            state.load(args.restore_ckpt, rank)

        while total_steps < args.num_steps:
            for i_batch, data_blob in enumerate(pbar:=tqdm(train_loader)):
                # if(pass_num>0):
                #     pass_num -= 1
                #     continue
                optimizer.zero_grad()
                depth_image, flow, valid = [x.cuda() for x in data_blob]
                with torch.no_grad():
                    flow_predictions, teach_feat, stud_feat = AsymKD_Compress.forward_with_inter_feat(depth_image)
                    knowledge_feat = teach_feat - stud_feat # shape: B, hw, 384
                    # knowledge_feat = (knowledge_feat - knowledge_feat.mean(dim=-1,keepdim=True)) / knowledge_feat.std(dim=-1,keepdim=True)
                    # knowledge_feat = rearrange(knowledge_feat, 'b n (i k) -> b n i k', i=12)
                    # knowledge_feat = knowledge_feat[:, :, 0, :] # shape: B, hw, 32
                    # print(f"teach_feat norm: {teach_feat.norm(dim=-1).mean()}")
                    # print(f"teach_feat mean: {teach_feat.mean(dim=-1).mean()}")
                    # print(f"teach_feat std: {teach_feat.std(dim=-1).mean()}")
                    # print(f"stud_feat norm: {stud_feat.norm(dim=-1).mean()}")
                    # print(f"stud_feat std: {stud_feat.mean(dim=-1).mean()}")
                    # print(f"stud_feat std: {stud_feat.std(dim=-1).mean()}")
                    # print(f"knowledge_feat norm: {knowledge_feat.norm(dim=-1).mean()}")
                    # print(f"knowledge_feat mean: {knowledge_feat.mean()}")
                    # print(f"knowledge_feat std: {knowledge_feat.std(dim=-1).mean()}")
                    # assert 0

                assert model.training
                loss = model(target=knowledge_feat, cond=stud_feat)
                assert model.training
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                pbar.set_description(f"diff_loss: {loss.item():.4f}")
                if rank == 0:
                    logger.writer.add_scalar("diff_loss", loss.item(), total_steps)
                    logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], total_steps)
                    _ , metrics = sequence_loss(flow_predictions, flow, valid)
                    logger.push(metrics)

                    if total_steps % 100 == 99:
                        # inference visualization in tensorboard while training
                        rgb = depth_image[0].cpu().detach().numpy()
                        rgb = ((rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))) * 255
            
                        gt = flow[0].cpu().detach().numpy()
                        gt = ((gt - np.min(gt)) / (np.max(gt) - np.min(gt))) * 255
            
                        pred = flow_predictions[0].cpu().detach().numpy()
                        pred = ((pred - np.min(pred)) / (np.max(pred) - np.min(pred))) * 255
            
                        logger.writer.add_image('RGB', rgb.astype(np.uint8), total_steps)
                        logger.writer.add_image('GT', gt.astype(np.uint8), total_steps)
                        logger.writer.add_image('Prediction', pred.astype(np.uint8), total_steps)

                    if total_steps % save_step == save_step-1:
                        save_path = Path(f"checkpoint_v_depth_latent1_avg_ver_stage_2/{total_steps + 1}_{args.name}.pth")
                        logging.info(f"Saving file {save_path.absolute()}")
                        state.save(save_path)
                
                if rank == 0 and total_steps % 100 == 99:
                    model.eval()
                    with torch.no_grad():
                        h, w = depth_image.shape[-2:]
                        cond_feat = stud_feat[0,...].unsqueeze(0)
                        cond_feat = rearrange(cond_feat, 'b n c -> (b n) c')
                        knowledge_feat = model.module.sample(cond_feat, batch_size=1)
                        knowledge_feat = rearrange(knowledge_feat, '(b n) c -> b n c', b=1)
                        compress_feat = stud_feat + knowledge_feat
                        depth = AsymKD_Compress.pred_dep_with_compress_feat(compress_feat, h, w)
                        depth = depth[0].cpu().detach().numpy()
                        depth = ((depth - np.min(depth)) / (np.max(depth) - np.min(depth))) * 255
                        # flow = vutils.make_grid([depth, gt] , nrow=1, normalize=True, scale_each=True)
                        logger.writer.add_image('Genearation', depth.astype(np.uint8), total_steps)
                    model.train()

                if total_steps%100==0:
                    torch.cuda.empty_cache()
                    gc.collect()

                total_steps += 1
            epoch += 1      

        print("FINISHED TRAINING")
        logger.close()
        state.save(f"checkpoint_v_depth_latent1_avg_ver_stage_2/{args.name}.pth")
        return None
    finally:
        cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='AsymKD_new_loss', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--epoch', type=int, default=3, help="length of training schedule.")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['tartan_air'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.00005, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[518, 518], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['hf','h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    Path("checkpoint_v_depth_latent1_avg_ver_stage_2").mkdir(exist_ok=True, parents=True)
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,args,), nprocs=world_size, join=True)
