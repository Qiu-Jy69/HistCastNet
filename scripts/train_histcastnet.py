# Derived from Earthformer (Apache License 2.0).
# Modified for HistCastNet.
import warnings
from typing import Union, Dict
from shutil import copyfile
from copy import deepcopy
import inspect
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
    Callback,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from pytorch_lightning import Trainer, seed_everything
from histcastnet.config import cfg
from histcastnet.utils.optim import SequentialLR, warmup_lambda
from histcastnet.utils.utils import get_parameter_names
from histcastnet.utils.checkpoint import pl_ckpt_to_pytorch_state_dict
from histcastnet.utils.layout import layout_to_in_out_slice
from histcastnet.visualization.sevir.sevir_vis_seq import save_example_vis_results
from histcastnet.metrics.sevir import SEVIRSkillScore
from sevir_csv_results import ScalarMeanMetric, save_sevir_test_results_csv
from histcastnet.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
from histcastnet.utils.apex_ddp import ApexDDPStrategy
from histcastnet.layers.DWT_IDWT_layer import FrameWiseDWT2D, FrameWiseIDWT2D


from histcastnet.models import HistCastNetBackbone, HistCastNet, PixelDualAttModel

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")
pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir
pytorch_state_dict_name = "histcastnet_sevir.pt"


class DynamicWeightedLoss(nn.Module):
    def __init__(self, extreme_weight=10.0):
        super().__init__()
        self.extreme_weight = extreme_weight

    def forward(self, pred, target):

        mse = F.mse_loss(pred, target, reduction="none")
        l1 = F.l1_loss(pred, target, reduction="none")

        weight_mask = 1.0 + self.extreme_weight * (target**2)

        loss = (0.5 * mse + 0.5 * l1) * weight_mask
        return loss.mean()


class NeighborhoodLoss(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=window_size, stride=1, padding=window_size // 2
        )

    def forward(self, pred, target):
        B, T, H, W, C = pred.shape
        p = pred.reshape(B * T, C, H, W)
        t = target.reshape(B * T, C, H, W)

        p_pool = self.pool(p)
        t_pool = self.pool(t)

        return F.l1_loss(p_pool, t_pool)


def compute_fss_batch(
    pred: torch.Tensor, target: torch.Tensor, threshold: float, window_size: int
) -> torch.Tensor:

    pred_event = (pred >= threshold).float()
    target_event = (target >= threshold).float()

    B, T, H, W, C = pred_event.shape
    p_flat = pred_event.reshape(B * T, C, H, W)
    t_flat = target_event.reshape(B * T, C, H, W)

    pad = window_size // 2
    pool = nn.AvgPool2d(kernel_size=window_size, stride=1, padding=pad)

    p_frac = pool(p_flat)
    t_frac = pool(t_flat)

    mse = torch.mean((p_frac - t_frac) ** 2, dim=(-1, -2))
    ref = torch.mean(p_frac**2 + t_frac**2, dim=(-1, -2))

    fss = 1.0 - (mse / (ref + 1e-6))

    return fss.mean()


class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.2):
        super().__init__()
        self.alpha = alpha

        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        self.register_buffer("kernel", kernel.reshape(1, 1, 3, 3))

    def forward(self, pred, target):

        x = pred.squeeze(-1)
        y = target.squeeze(-1)
        B, T, H, W = x.shape

        x = x.reshape(B * T, 1, H, W)
        y = y.reshape(B * T, 1, H, W)
        k = self.kernel
        x_hf = F.conv2d(x, k, padding=1)
        y_hf = F.conv2d(y, k, padding=1)
        diff = torch.abs(x_hf - y_hf)
        loss = (diff**self.alpha).mean()
        return loss


class EventFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, thr: float, temp: float
    ) -> torch.Tensor:

        #

        temp = max(float(temp), self.eps)
        target_event = (target >= thr).float()
        logits = (pred - thr) / temp
        bce = F.binary_cross_entropy_with_logits(logits, target_event, reduction="none")
        pred_prob = torch.sigmoid(logits)
        pt = pred_prob * target_event + (1.0 - pred_prob) * (1.0 - target_event)
        alpha_t = self.alpha * target_event + (1.0 - self.alpha) * (1.0 - target_event)
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        return loss.mean()


class WaveletLoss(nn.Module):
    def __init__(self, wavename="haar"):
        super().__init__()
        self.dwt = FrameWiseDWT2D(wavename=wavename)

    def forward(self, pred, target):

        #

        with autocast(enabled=False):
            pred_dwt = self.dwt(pred.float())
            target_dwt = self.dwt(target.float())
        pred_hf = pred_dwt[..., 1:]
        target_hf = target_dwt[..., 1:]
        loss = F.l1_loss(pred_hf, target_hf)
        return loss


class HistCastNetModule(pl.LightningModule):

    def __init__(self, total_num_steps: int, oc_file: str = None, save_dir: str = None):
        super(HistCastNetModule, self).__init__()

        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        model_cfg = OmegaConf.to_object(oc.model)

        use_wavelet_backbone = bool(model_cfg.get("use_wavelet_backbone", True))

        if use_wavelet_backbone:

            height = self.oc.dataset.img_height
            width = self.oc.dataset.img_width
            in_len = self.oc.layout.in_len
            out_len = self.oc.layout.out_len
            data_channels = 1
            H2 = height // 2
            W2 = width // 2
            C_wave = 4 * data_channels

            input_shape = (in_len, H2, W2, C_wave)
            target_shape = (out_len, H2, W2, C_wave)
            model_cfg["input_shape"] = input_shape
            model_cfg["target_shape"] = target_shape

            inner_model = HistCastNetBackbone(
                input_shape=input_shape,
                target_shape=target_shape,
                base_dim=model_cfg.get("base_units", 64),
                stage1_dim=model_cfg.get("stage1_dim", 128),
                num_blocks_stage0=model_cfg.get("num_blocks_stage0", 2),
                num_blocks_stage1=model_cfg.get("num_blocks_stage1", 2),
                num_heads=model_cfg.get("num_heads", 4),
                window_size=model_cfg.get("window_size", (4, 4)),
                attn_drop=model_cfg.get("attn_drop", 0.1),
                ffn_drop=model_cfg.get("ffn_drop", 0.1),
                use_temporal_attn=model_cfg.get("use_temporal_attn", True),
                use_spatial_attn=model_cfg.get("use_spatial_attn", True),
                fshrd_variant=model_cfg.get("fshrd_variant", "full"),
            )

            self.torch_nn_module = HistCastNet(
                inner_model=inner_model,
                dwt_layer=FrameWiseDWT2D(wavename="haar"),
                idwt_layer=FrameWiseIDWT2D(wavename="haar"),
            )
        else:

            height = self.oc.dataset.img_height
            width = self.oc.dataset.img_width
            in_len = self.oc.layout.in_len
            out_len = self.oc.layout.out_len
            data_channels = 1

            input_shape = (in_len, height, width, data_channels)
            target_shape = (out_len, height, width, data_channels)
            model_cfg["input_shape"] = input_shape
            model_cfg["target_shape"] = target_shape

            inner_model = PixelDualAttModel(
                input_shape=input_shape,
                target_shape=target_shape,
                base_dim=model_cfg.get("base_units", 64),
                stage1_dim=model_cfg.get("stage1_dim", 128),
                num_blocks_stage0=model_cfg.get("num_blocks_stage0", 2),
                num_blocks_stage1=model_cfg.get("num_blocks_stage1", 2),
                num_heads=model_cfg.get("num_heads", 4),
                window_size=model_cfg.get("window_size", (4, 4)),
                attn_drop=model_cfg.get("attn_drop", 0.1),
                ffn_drop=model_cfg.get("ffn_drop", 0.1),
                use_temporal_attn=model_cfg.get("use_temporal_attn", True),
                use_spatial_attn=model_cfg.get("use_spatial_attn", True),
            )
            self.torch_nn_module = inner_model

        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.configure_save(cfg_file_path=oc_file)

        self.ffl = FocalFrequencyLoss(alpha=1.2)
        self.event_focal_loss = EventFocalLoss()
        self.wavelet_loss_func = WaveletLoss(wavename="haar")

        event_cfg = model_cfg.get("event_loss", {})
        self.event_loss_enable = bool(event_cfg.get("enable", True))
        self.event_thrs_raw = event_cfg.get("thr_raw_list", [133, 160, 181, 219])
        self.event_thrs = [float(t) / 255.0 for t in self.event_thrs_raw]
        self.event_thr_weights = event_cfg.get("thr_weight_list", [0.2, 0.3, 0.3, 0.2])
        self.event_temps = event_cfg.get("temp_list", [0.03, 0.02, 0.02, 0.01])
        self.event_alpha = float(event_cfg.get("alpha", 0.55))
        self.event_gamma = float(event_cfg.get("gamma", 2.0))
        self.event_focal_loss.alpha = self.event_alpha
        self.event_focal_loss.gamma = self.event_gamma

        loss_weights_cfg = model_cfg.get("loss_weights", {})
        renormalize = bool(loss_weights_cfg.get("renormalize", True))

        w_base = float(loss_weights_cfg.get("base", 0.45))
        w_event = float(loss_weights_cfg.get("event", 0.40))
        w_ffl = float(loss_weights_cfg.get("ffl", 0.15))
        w_wavelet = float(loss_weights_cfg.get("wavelet", 0.0))

        w_sum = w_base + w_wavelet + w_event + w_ffl
        if renormalize and w_sum > 0:

            w_base /= w_sum
            w_wavelet /= w_sum
            w_event /= w_sum
            w_ffl /= w_sum

        self.loss_weight_base = w_base
        self.loss_weight_wavelet = w_wavelet
        self.loss_weight_event = w_event
        self.loss_weight_ffl = w_ffl

        self.metrics_list = oc.dataset.metrics_list
        self.threshold_list = oc.dataset.threshold_list
        self.metrics_mode = oc.dataset.metrics_mode
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.valid_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,
        )
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,
        )
        self.test_frame_score = SEVIRSkillScore(
            mode="1",
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,
        )
        self.test_fss_score = nn.ModuleDict()
        for thr_raw in self.threshold_list:
            self.test_fss_score[f"{int(thr_raw)}_w5"] = ScalarMeanMetric()
            self.test_fss_score[f"{int(thr_raw)}_w11"] = ScalarMeanMetric()

        self.dynamic_loss = DynamicWeightedLoss(extreme_weight=10.0)

        self.neighbor_loss_func = NeighborhoodLoss(window_size=5)

    def configure_save(self, cfg_file_path=None):

        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, "scores")
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or (
                not os.path.samefile(cfg_file_path, cfg_file_target_path)
            ):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):

        oc = OmegaConf.create()
        oc.dataset = self.get_dataset_config()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_dataset_config():
        oc = OmegaConf.create()
        oc.dataset_name = "sevir_lr"
        oc.img_height = 128
        oc.img_width = 128
        oc.in_len = 7
        oc.out_len = 6
        oc.seq_len = 13
        oc.plot_stride = 1
        oc.interval_real_time = 10
        oc.sample_mode = "sequent"
        oc.stride = oc.out_len
        oc.layout = "NTHWC"
        oc.num_workers = 24
        oc.use_fixed_interim_split = True
        oc.start_date = None
        oc.train_val_split_date = (2019, 1, 1)
        oc.train_test_split_date = (2019, 6, 1)
        oc.end_date = None
        oc.metrics_mode = "0"
        oc.metrics_list = ("csi", "pod", "sucr", "bias", "ets")
        oc.threshold_list = (16, 74, 133, 160, 181, 219)
        return oc

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        height = dataset_oc.img_height
        width = dataset_oc.img_width
        in_len = dataset_oc.in_len
        out_len = dataset_oc.out_len
        data_channels = 1
        cfg.use_wavelet_backbone = True

        if cfg.use_wavelet_backbone:
            H2, W2, C_wave = height // 2, width // 2, 4 * data_channels
            cfg.input_shape = (in_len, H2, W2, C_wave)
            cfg.target_shape = (out_len, H2, W2, C_wave)
        else:
            cfg.input_shape = (in_len, height, width, data_channels)
            cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.stage1_dim = 128
        cfg.num_blocks_stage0 = 2
        cfg.num_blocks_stage1 = 2
        cfg.num_heads = 4
        cfg.window_size = [4, 4]
        cfg.attn_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.use_temporal_attn = True
        cfg.use_spatial_attn = True
        cfg.fshrd_variant = "full"

        cfg.event_loss = {
            "enable": True,
            "thr_raw_list": [133, 160, 181, 219],
            "thr_weight_list": [0.2, 0.3, 0.3, 0.2],
            "temp_list": [0.03, 0.02, 0.02, 0.01],
            "gamma": 2.0,
            "alpha": 0.55,
        }
        cfg.loss_weights = {
            "base": 0.45,
            "event": 0.40,
            "ffl": 0.15,
            "wavelet": 0.0,
            "renormalize": True,
        }
        return cfg

    @classmethod
    def get_layout_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.in_len = dataset_oc.in_len
        oc.out_len = dataset_oc.out_len
        oc.layout = dataset_oc.layout
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 128
        oc.micro_batch_size = 16
        oc.method = "adamw"
        oc.lr = 1e-3
        oc.wd = 1e-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 100
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"
        oc.min_lr_ratio = 0.001
        oc.warmup_min_lr_ratio = 0.0
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 20
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "SEVIR"
        oc.monitor_lr = True
        oc.monitor_device = False
        oc.track_grad_norm = -1
        oc.use_wandb = True
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001
        oc.precision = 32
        return oc

    @classmethod
    def get_vis_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.train_example_data_idx_list = []
        oc.val_example_data_idx_list = []
        oc.test_example_data_idx_list = [0, 80, 160, 240, 320, 400]
        oc.eval_example_only = False
        oc.plot_stride = dataset_oc.plot_stride
        oc.save_individual_frames = False
        return oc

    def configure_optimizers(self):

        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.torch_nn_module.named_parameters()
                    if n in decay_parameters
                ],
                "weight_decay": self.oc.optim.wd,
            },
            {
                "params": [
                    p
                    for n, p in self.torch_nn_module.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.oc.optim.method == "adamw":
            optimizer = torch.optim.AdamW(
                params=optimizer_grouped_parameters,
                lr=self.oc.optim.lr,
                weight_decay=self.oc.optim.wd,
            )
        else:
            raise NotImplementedError

        warmup_iter = int(
            np.round(self.oc.optim.warmup_percentage * self.total_num_steps)
        )

        if self.oc.optim.lr_scheduler_mode == "cosine":

            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=warmup_lambda(
                    warmup_steps=warmup_iter,
                    min_lr_ratio=self.oc.optim.warmup_min_lr_ratio,
                ),
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=(self.total_num_steps - warmup_iter),
                eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr,
            )
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_iter],
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise NotImplementedError
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):

        best_loss_checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="best_loss-epoch{epoch:03d}-avgcsi{valid_csi_avg_epoch:.4f}",
            save_top_k=1,
            save_last=False,
            mode="min",
        )
        best_ext_checkpoint_callback = ModelCheckpoint(
            monitor="valid_ext_score_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="best_ext-epoch{epoch:03d}-ext{valid_ext_score_epoch:.4f}",
            save_top_k=1,
            save_last=False,
            mode="max",
        )
        last_checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="last-epoch{epoch:03d}",
            save_top_k=0,
            save_last=True,
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [
            best_loss_checkpoint_callback,
            best_ext_checkpoint_callback,
            last_checkpoint_callback,
        ]
        if self.oc.logging.monitor_lr:
            callbacks += [
                LearningRateMonitor(logging_interval="step"),
            ]
        if self.oc.logging.monitor_device:
            callbacks += [
                DeviceStatsMonitor(),
            ]
        if self.oc.optim.early_stop:
            callbacks += [
                EarlyStopping(
                    monitor="valid_loss_epoch",
                    min_delta=0.0,
                    patience=self.oc.optim.early_stop_patience,
                    verbose=False,
                    mode=self.oc.optim.early_stop_mode,
                ),
            ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(
                project=self.oc.logging.logging_prefix, save_dir=self.save_dir
            )
            logger += [
                wandb_logger,
            ]

        log_every_n_steps = max(
            1, int(self.oc.trainer.log_step_ratio * self.total_num_steps)
        )
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            track_grad_norm=self.oc.logging.track_grad_norm,
            default_root_dir=self.save_dir,
            accelerator="gpu",
            strategy=ApexDDPStrategy(find_unused_parameters=True, delay_allreduce=True),
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {
            key: val
            for key, val in oc_trainer_kwargs.items()
            if key in trainer_init_keys
        }
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
        cls, num_samples: int, total_batch_size: int, epoch: int = None
    ):
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_sevir_datamodule(
        dataset_oc, micro_batch_size: int = 1, num_workers: int = 8
    ):
        dm = SEVIRLightningDataModule(
            seq_len=dataset_oc["seq_len"],
            sample_mode=dataset_oc["sample_mode"],
            stride=dataset_oc["stride"],
            batch_size=micro_batch_size,
            layout=dataset_oc["layout"],
            output_type=np.float32,
            preprocess=True,
            rescale_method="01",
            verbose=False,
            dataset_name=dataset_oc["dataset_name"],
            start_date=dataset_oc["start_date"],
            train_val_split_date=dataset_oc["train_val_split_date"],
            train_test_split_date=dataset_oc["train_test_split_date"],
            end_date=dataset_oc["end_date"],
            use_fixed_interim_split=dataset_oc.get("use_fixed_interim_split", True),
            num_workers=num_workers,
        )
        return dm

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.layout, in_len=self.in_len, out_len=self.out_len
            )
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.layout, in_len=self.in_len, out_len=self.out_len
            )
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    def multi_threshold_event_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:

        total = pred.new_zeros(())
        weight_sum = 0.0
        for thr, w, temp in zip(
            self.event_thrs, self.event_thr_weights, self.event_temps
        ):
            loss_k = self.event_focal_loss(
                pred=pred,
                target=target,
                thr=thr,
                temp=temp,
            )
            total = total + float(w) * loss_k
            weight_sum += float(w)
        if weight_sum > 0:
            total = total / weight_sum
        return total

    def forward(self, in_seq, out_seq=None):
        output = self.torch_nn_module(in_seq)

        if out_seq is None:
            return output

        pixel_loss = self.dynamic_loss(output, out_seq)

        neighbor_loss = self.neighbor_loss_func(output, out_seq)

        base_loss = 0.5 * pixel_loss + 0.5 * neighbor_loss

        return output, base_loss

    def training_step(self, batch, batch_idx):

        data_seq = batch["vil"].contiguous()
        x = data_seq[self.in_slice]
        y = data_seq[self.out_slice]

        y_hat, base_loss = self(x, y)
        self.log("train_base_loss", base_loss, on_step=True, on_epoch=False)

        pred = y_hat
        target = y
        if pred.shape != target.shape:

            target = target.reshape(pred.shape)

        if self.loss_weight_ffl > 0:
            ffl_loss = self.ffl(pred, target)
        else:
            ffl_loss = pred.new_zeros(())
        self.log("train_ffl_loss", ffl_loss, on_step=True, on_epoch=False)

        if self.loss_weight_wavelet > 0:
            wavelet_loss = self.wavelet_loss_func(pred, target)
        else:
            wavelet_loss = pred.new_zeros(())
        self.log("train_wavelet_loss", wavelet_loss, on_step=True, on_epoch=False)

        if self.event_loss_enable and self.loss_weight_event > 0:
            event_loss = self.multi_threshold_event_loss(pred=pred, target=target)
        else:
            event_loss = pred.new_zeros(())
        self.log("train_event_loss", event_loss, on_step=True, on_epoch=False)

        loss = (
            self.loss_weight_base * base_loss
            + self.loss_weight_event * event_loss
            + self.loss_weight_ffl * ffl_loss
            + self.loss_weight_wavelet * wavelet_loss
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite train_loss detected at epoch={self.current_epoch}, batch_idx={batch_idx}."
            )

        micro_batch_size = x.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        self.save_vis_step_end(
            data_idx=data_idx, in_seq=x, target_seq=y, pred_seq=y_hat, mode="train"
        )
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):

        data_seq = batch["vil"].contiguous()
        x = data_seq[self.in_slice]
        y = data_seq[self.out_slice]
        micro_batch_size = x.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            y_hat, _ = self(x, y)
            if self.precision == 16:
                y_hat = y_hat.float()
            step_mse = self.valid_mse(y_hat, y)
            step_mae = self.valid_mae(y_hat, y)
            self.valid_score.update(y_hat, y)
            self.log(
                "valid_frame_mse_step",
                step_mse,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "valid_frame_mae_step",
                step_mae,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
        return None

    def validation_epoch_end(self, outputs):

        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        self.log(
            "valid_frame_mse_epoch",
            valid_mse,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "valid_frame_mae_epoch",
            valid_mae,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.valid_mse.reset()
        self.valid_mae.reset()

        valid_score = self.valid_score.compute()

        valid_avg_csi = np.mean(valid_score["avg"]["csi"]).item()
        self.log(
            "valid_loss_epoch",
            -valid_avg_csi,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        valid_csi_160 = np.mean(valid_score[160]["csi"]).item()
        valid_csi_181 = np.mean(valid_score[181]["csi"]).item()
        valid_csi_219 = np.mean(valid_score[219]["csi"]).item()
        valid_sucr_181 = np.mean(valid_score[181]["sucr"]).item()

        self.log(
            "valid_csi_160_epoch_manual", valid_csi_160, on_step=False, on_epoch=True
        )
        self.log(
            "valid_csi_181_epoch_manual", valid_csi_181, on_step=False, on_epoch=True
        )
        self.log(
            "valid_csi_219_epoch_manual", valid_csi_219, on_step=False, on_epoch=True
        )
        self.log(
            "valid_sucr_181_epoch_manual", valid_sucr_181, on_step=False, on_epoch=True
        )

        valid_ext_score = (
            0.5 * valid_csi_160 + 0.3 * valid_csi_181 + 0.2 * valid_csi_219
        )
        self.log(
            "valid_ext_score_epoch",
            valid_ext_score,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self.log_score_epoch_end(score_dict=valid_score, mode="val")
        self.valid_score.reset()
        self.save_score_epoch_end(
            score_dict=valid_score, mse=valid_mse, mae=valid_mae, mode="val"
        )

    def test_step(self, batch, batch_idx):
        data_seq = batch["vil"].contiguous()
        x = data_seq[self.in_slice]
        y = data_seq[self.out_slice]
        micro_batch_size = x.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            y_hat, _ = self(x, y)
            self.save_vis_step_end(
                data_idx=data_idx, in_seq=x, target_seq=y, pred_seq=y_hat, mode="test"
            )
            if self.precision == 16:
                y_hat = y_hat.float()
            step_mse = self.test_mse(y_hat, y)
            step_mae = self.test_mae(y_hat, y)
            self.test_score.update(y_hat, y)
            self.test_frame_score.update(y_hat, y)
            self.log(
                "test_frame_mse_step",
                step_mse,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "test_frame_mae_step",
                step_mae,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )

            for thr_raw in self.threshold_list:
                thr_norm = thr_raw / 255.0
                fss_w5 = compute_fss_batch(y_hat, y, threshold=thr_norm, window_size=5)
                fss_w11 = compute_fss_batch(
                    y_hat, y, threshold=thr_norm, window_size=11
                )
                self.test_fss_score[f"{int(thr_raw)}_w5"].update(fss_w5)
                self.test_fss_score[f"{int(thr_raw)}_w11"].update(fss_w11)

        return None

    def test_epoch_end(self, outputs):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        self.log(
            "test_frame_mse_epoch",
            test_mse,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_frame_mae_epoch",
            test_mae,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.test_mse.reset()
        self.test_mae.reset()
        test_score = self.test_score.compute()
        test_frame_score = self.test_frame_score.compute()
        test_extra_score = {}
        for thr_raw in self.threshold_list:
            test_extra_score[int(thr_raw)] = {
                "fss_w5": self.test_fss_score[f"{int(thr_raw)}_w5"].compute(),
                "fss_w11": self.test_fss_score[f"{int(thr_raw)}_w11"].compute(),
            }
        self.test_score.reset()
        self.test_frame_score.reset()
        for metric in self.test_fss_score.values():
            metric.reset()
        csv_path = self.save_score_epoch_end(
            score_dict=test_score,
            mse=test_mse,
            mae=test_mae,
            mode="test",
            frame_score_dict=test_frame_score,
            extra_score_dict=test_extra_score,
        )
        if csv_path is not None:
            print(f"Test metrics saved to: {csv_path}")

    def log_score_epoch_end(self, score_dict: Dict, mode: str = "val"):
        if mode == "val":
            log_mode_prefix = "valid"
        elif mode == "test":
            log_mode_prefix = "test"
        else:
            raise ValueError(f"Wrong mode {mode}. Must be 'val' or 'test'.")
        for metrics in self.metrics_list:
            for thresh in self.threshold_list:
                score_mean = np.mean(score_dict[thresh][metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_{thresh}_epoch", score_mean)
            score_avg_mean = score_dict.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_avg_epoch", score_avg_mean)

    def save_score_epoch_end(
        self,
        score_dict: Dict,
        mse: Union[np.ndarray, float],
        mae: Union[np.ndarray, float],
        mode: str = "val",
        frame_score_dict: Dict = None,
        extra_score_dict: Dict = None,
    ):
        assert mode in ["val", "test"], f"Wrong mode {mode}. Must be 'val' or 'test'."
        if self.local_rank == 0 and self.scores_dir is not None:
            if mode == "test":
                interval_real_time = int(self.oc.dataset.get("interval_real_time", 10))
                return save_sevir_test_results_csv(
                    scores_dir=self.scores_dir,
                    epoch=self.current_epoch,
                    score_dict=score_dict,
                    frame_score_dict=frame_score_dict,
                    extra_score_dict=extra_score_dict,
                    mse=mse,
                    mae=mae,
                    metrics_list=self.metrics_list,
                    threshold_list=self.threshold_list,
                    interval_real_time=interval_real_time,
                )

            save_dict = deepcopy(score_dict)
            save_dict.update(dict(mse=mse, mae=mae))
            save_path = os.path.join(
                self.scores_dir, f"{mode}_results_epoch_{self.current_epoch}.pkl"
            )
            with open(save_path, "wb") as f:
                pickle.dump(save_dict, f)
            return save_path
        return None

    def save_vis_step_end(
        self,
        data_idx: int,
        in_seq: torch.Tensor,
        target_seq: torch.Tensor,
        pred_seq: torch.Tensor,
        mode: str = "train",
    ):
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(
                    f"Wrong mode {mode}! Must be in ['train', 'val', 'test']."
                )
            if data_idx in example_data_idx_list:
                save_example_vis_results(
                    save_dir=self.example_save_dir,
                    save_prefix=f"{mode}_epoch_{self.current_epoch}_data_{data_idx}",
                    in_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    layout=self.layout,
                    plot_stride=self.oc.vis.plot_stride,
                    label=self.oc.logging.logging_prefix,
                    interval_real_time=self.oc.dataset.interval_real_time,
                    save_individual_frames=self.oc.vis.get(
                        "save_individual_frames", False
                    ),
                )


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="tmp_sevir", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--cfg", default=None, type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained checkpoints for test.",
    )
    parser.add_argument(
        "--ckpt_name",
        default=None,
        type=str,
        help="The model checkpoint trained on SEVIR.",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.pretrained:
        if args.ckpt_name is None:
            parser.error(
                "--pretrained requires --ckpt_name pointing to a local state dict"
            )
        if args.cfg is None:
            args.cfg = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "configs",
                    "histcastnet_sevirlr.yaml",
                )
            )
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_oc = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        dataset_oc = OmegaConf.to_object(HistCastNetModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0

    seed_everything(seed, workers=True)
    num_workers = dataset_oc.get("num_workers", 24)
    dm = HistCastNetModule.get_sevir_datamodule(
        dataset_oc=dataset_oc,
        micro_batch_size=micro_batch_size,
        num_workers=num_workers,
    )
    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    if total_batch_size % (micro_batch_size * args.gpus) != 0:
        raise ValueError(
            f"total_batch_size={total_batch_size} must be divisible by "
            f"micro_batch_size({micro_batch_size}) * gpus({args.gpus})."
        )
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    effective_total_batch_size = micro_batch_size * args.gpus * accumulate_grad_batches
    print(f"total_batch_size = {total_batch_size}")
    print(f"micro_batch_size = {micro_batch_size}")
    print(f"world_size = {args.gpus}")
    print(f"gradient accumulation = {accumulate_grad_batches}")
    print(f"effective total batch size = {effective_total_batch_size}")

    total_num_steps = HistCastNetModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = HistCastNetModule(
        total_num_steps=total_num_steps, save_dir=args.save, oc_file=args.cfg
    )
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)

    if args.pretrained:
        # --pretrained loads a user-supplied local PyTorch state dict.
        checkpoint_path = os.path.abspath(args.ckpt_name)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict, strict=False)
        trainer.test(model=pl_module, datamodule=dm)
    elif args.test:

        assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
    else:

        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(
                    f"ckpt {ckpt_path} not exists! Start training from epoch 0."
                )
                ckpt_path = None
        else:
            ckpt_path = None

        trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)

        ckpt_dir = os.path.join(pl_module.save_dir, "checkpoints")
        best_ext_candidates = (
            sorted(
                [
                    os.path.join(ckpt_dir, f)
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("best_ext-") and f.endswith(".ckpt")
                ]
            )
            if os.path.isdir(ckpt_dir)
            else []
        )
        best_loss_candidates = (
            sorted(
                [
                    os.path.join(ckpt_dir, f)
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("best_loss-") and f.endswith(".ckpt")
                ]
            )
            if os.path.isdir(ckpt_dir)
            else []
        )

        export_ckpt_path = (
            best_ext_candidates[-1]
            if len(best_ext_candidates) > 0
            else (best_loss_candidates[-1] if len(best_loss_candidates) > 0 else None)
        )

        if export_ckpt_path and os.path.isfile(export_ckpt_path):
            state_dict = pl_ckpt_to_pytorch_state_dict(
                checkpoint_path=export_ckpt_path,
                map_location=torch.device("cpu"),
                delete_prefix_len=len("torch_nn_module."),
            )
            # Non-model Lightning entries (currently ffl.kernel) do not carry
            # the torch_nn_module prefix and previously became an empty key.
            state_dict.pop("", None)
            torch.save(
                state_dict,
                os.path.join(
                    pl_module.save_dir, "checkpoints", pytorch_state_dict_name
                ),
            )
            print(
                f">>> Training complete. Best checkpoint exported to {pytorch_state_dict_name}."
            )
            print(">>> Run with --test to evaluate the exported checkpoint.")
        else:
            warnings.warn("No best checkpoint was found for export.")


if __name__ == "__main__":
    main()
