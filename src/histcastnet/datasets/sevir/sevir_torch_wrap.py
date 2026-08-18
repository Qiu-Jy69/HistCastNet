# Derived from Earthformer (Apache License 2.0).
# Modified for HistCastNet.
import os
from typing import Union, Dict, Sequence, Tuple, List
import numpy as np
import datetime
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from pytorch_lightning import LightningDataModule
from ...config import cfg
from .sevir_dataloader import SEVIRDataLoader
from ...utils.layout import change_layout_np


class SEVIRTorchDataset(TorchDataset):

    def __init__(
        self,
        seq_len: int = 25,
        raw_seq_len: int = 49,
        sample_mode: str = "sequent",
        stride: int = 12,
        batch_size: int = 1,
        layout: str = "NHWT",
        num_shard: int = 1,
        rank: int = 0,
        split_mode: str = "uneven",
        sevir_catalog: Union[str, pd.DataFrame] = None,
        sevir_data_dir: str = None,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        datetime_filter=None,
        catalog_filter="default",
        shuffle: bool = False,
        shuffle_seed: int = 1,
        output_type=np.float32,
        preprocess: bool = True,
        rescale_method: str = "01",
        verbose: bool = False,
    ):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=[
                "vil",
            ],
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=batch_size,
            layout=layout,
            num_shard=num_shard,
            rank=rank,
            split_mode=split_mode,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            rescale_method=rescale_method,
            downsample_dict=None,
            verbose=verbose,
        )

    def __getitem__(self, index):
        data_dict = self.sevir_dataloader._idx_sample(index=index)
        return data_dict

    def __len__(self):
        return self.sevir_dataloader.__len__()

    def collate_fn(self, data_dict_list):
        r"""
        Parameters
        ----------
        data_dict_list:  list[Dict[str, torch.Tensor]]

        Returns
        -------
        merged_data: Dict[str, torch.Tensor]
            batch_size = len(data_dict_list) * data_dict["key"].batch_size
        """
        batch_dim = self.layout.find("N")
        data_list_dict = {
            key: [data_dict[key] for data_dict in data_dict_list]
            for key in data_dict_list[0]
        }
        # This helper does not merge masks.
        data_list_dict.pop("mask", None)
        merged_dict = {
            key: torch.cat(data_list, dim=batch_dim)
            for key, data_list in data_list_dict.items()
        }
        merged_dict["mask"] = None
        return merged_dict

    def get_torch_dataloader(self, outer_batch_size=1, collate_fn=None, num_workers=1):
        r"""
        We set the batch_size in Dataset by default, so outer_batch_size should be 1.
        In this case, not using `collate_fn` can save time.
        """
        if outer_batch_size == 1:
            collate_fn = lambda x: x[0]
        else:
            if collate_fn is None:
                collate_fn = self.collate_fn
        dataloader = DataLoader(
            dataset=self,
            batch_size=outer_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        return dataloader


class SEVIRNowcastH5Dataset(TorchDataset):
    """Read fixed train/val/test splits from nowcast_*.h5 (IN/OUT)."""

    def __init__(
        self,
        h5_path: str,
        layout: str = "NTHWC",
        output_type=np.float32,
        preprocess: bool = True,
        rescale_method: str = "01",
    ):
        super().__init__()
        self.h5_path = h5_path
        self.layout = layout
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self._hf = None
        self._in = None
        self._out = None
        with h5py.File(self.h5_path, mode="r") as hf:
            self._len = int(hf["IN"].shape[0])

    def _ensure_open(self):
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, mode="r")
            self._in = self._hf["IN"]
            self._out = self._hf["OUT"]

    def __len__(self):
        return self._len

    def _preprocess(self, arr):
        arr = arr.astype(self.output_type, copy=False)
        if not self.preprocess:
            return arr
        if self.rescale_method == "01":
            arr = arr / 255.0
        elif self.rescale_method == "sevir":
            arr = (arr - 33.44) / 47.54
        else:
            raise ValueError(f"Unsupported rescale_method: {self.rescale_method}")
        return arr

    def __getitem__(self, index):
        self._ensure_open()
        in_seq = self._preprocess(self._in[index])  # (H, W, 7)
        out_seq = self._preprocess(self._out[index])  # (H, W, 6)
        seq_nhwt = np.concatenate([in_seq, out_seq], axis=-1)[
            None, ...
        ]  # (1, H, W, 13)
        seq = change_layout_np(seq_nhwt, in_layout="NHWT", out_layout=self.layout)[0]
        return {"vil": torch.from_numpy(seq), "mask": None}

    def get_torch_dataloader(self, outer_batch_size=1, collate_fn=None, num_workers=1):
        if collate_fn is None:
            collate_fn = lambda x: {
                "vil": torch.stack([ele["vil"] for ele in x], dim=0),
                "mask": None,
            }
        dataloader = DataLoader(
            dataset=self,
            batch_size=outer_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        return dataloader


def check_aws():
    r"""
    Check if aws cli is installed.
    """
    if os.system("which aws") != 0:
        raise RuntimeError(
            "AWS CLI is not installed! Please install it first. See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        )


def download_SEVIR(save_dir=None):
    r"""
    Downloaded dataset is saved in save_dir/sevir
    """

    check_aws()

    if save_dir is None:
        save_dir = cfg.datasets_dir
    sevir_dir = os.path.join(save_dir, "sevir")
    if os.path.exists(sevir_dir):
        raise FileExistsError(f"Path to save SEVIR dataset {sevir_dir} already exists!")
    else:
        os.makedirs(sevir_dir)
        os.system(
            f"aws s3 cp --no-sign-request s3://sevir/CATALOG.csv "
            f"{os.path.join(sevir_dir, 'CATALOG.csv')}"
        )
        os.system(
            f"aws s3 cp --no-sign-request --recursive s3://sevir/data/vil "
            f"{os.path.join(sevir_dir, 'data', 'vil')}"
        )


class SEVIRLightningDataModule(LightningDataModule):

    def __init__(
        self,
        seq_len: int = 25,
        sample_mode: str = "sequent",
        stride: int = 12,
        batch_size: int = 1,
        layout: str = "NHWT",
        output_type=np.float32,
        preprocess: bool = True,
        rescale_method: str = "01",
        verbose: bool = False,
        # datamodule_only
        dataset_name: str = "sevir",
        start_date: Tuple[int] = None,
        train_val_split_date: Tuple[int] = (2019, 1, 1),
        train_test_split_date: Tuple[int] = (2019, 6, 1),
        end_date: Tuple[int] = None,
        num_workers: int = 1,
        use_fixed_interim_split: bool = True,
    ):
        super(SEVIRLightningDataModule, self).__init__()
        self.seq_len = seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        self.layout = layout
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.num_workers = num_workers
        self.use_fixed_interim_split = use_fixed_interim_split
        if dataset_name == "sevir":
            sevir_root_dir = os.path.join(cfg.datasets_dir, "sevir")
            catalog_path = os.path.join(sevir_root_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_root_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 384
            img_width = 384
        elif dataset_name == "sevir_lr":
            sevir_root_dir = os.path.join(cfg.datasets_dir, "sevir_lr")
            catalog_path = os.path.join(sevir_root_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_root_dir, "data")
            interim_dir = os.path.join(sevir_root_dir, "interim")
            nowcast_train_path = os.path.join(interim_dir, "nowcast_train.h5")
            nowcast_val_path = os.path.join(interim_dir, "nowcast_val.h5")
            nowcast_test_path = os.path.join(interim_dir, "nowcast_test.h5")
            raw_seq_len = 25
            interval_real_time = 10
            img_height = 128
            img_width = 128
        else:
            raise ValueError(
                f"Wrong dataset name {dataset_name}. Must be 'sevir' or 'sevir_lr'."
            )
        self.dataset_name = dataset_name
        self.sevir_root_dir = sevir_root_dir
        self.catalog_path = catalog_path
        self.raw_data_dir = raw_data_dir
        self.nowcast_train_path = (
            nowcast_train_path if dataset_name == "sevir_lr" else None
        )
        self.nowcast_val_path = nowcast_val_path if dataset_name == "sevir_lr" else None
        self.nowcast_test_path = (
            nowcast_test_path if dataset_name == "sevir_lr" else None
        )
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width
        # train val test split
        self.start_date = (
            datetime.datetime(*start_date) if start_date is not None else None
        )
        self.train_val_split_date = datetime.datetime(*train_val_split_date)
        self.train_test_split_date = datetime.datetime(*train_test_split_date)
        self.end_date = datetime.datetime(*end_date) if end_date is not None else None

    def prepare_data(self) -> None:
        if (
            self.dataset_name == "sevir_lr"
            and self.use_fixed_interim_split
            and all(
                os.path.exists(path)
                for path in (
                    self.nowcast_train_path,
                    self.nowcast_val_path,
                    self.nowcast_test_path,
                )
            )
        ):
            return
        if os.path.exists(self.sevir_root_dir):
            # Further check
            assert os.path.exists(
                self.catalog_path
            ), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
            assert os.path.exists(
                self.raw_data_dir
            ), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        else:
            if self.dataset_name == "sevir":
                download_SEVIR()
            else:  # "sevir_lr"
                # raise NotImplementedError
                pass

    def setup(self, stage=None) -> None:
        use_nowcast_fixed_split = (
            self.use_fixed_interim_split
            and self.dataset_name == "sevir_lr"
            and self.nowcast_train_path is not None
            and os.path.exists(self.nowcast_train_path)
            and os.path.exists(self.nowcast_val_path)
            and os.path.exists(self.nowcast_test_path)
        )
        if use_nowcast_fixed_split:
            self.sevir_train = SEVIRNowcastH5Dataset(
                h5_path=self.nowcast_train_path,
                layout=self.layout,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
            )
            self.sevir_val = SEVIRNowcastH5Dataset(
                h5_path=self.nowcast_val_path,
                layout=self.layout,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
            )
            self.sevir_test = SEVIRNowcastH5Dataset(
                h5_path=self.nowcast_test_path,
                layout=self.layout,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
            )
            self.sevir_predict = self.sevir_test
            print(
                f">>> use fixed nowcast splits from: {os.path.dirname(self.nowcast_train_path)}"
            )
            print(">>> train samples =", len(self.sevir_train))
            print(">>> val samples   =", len(self.sevir_val))
            print(">>> test samples  =", len(self.sevir_test))
            return

        self.sevir_train = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=True,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            num_shard=1,
            rank=0,
            start_date=self.start_date,
            end_date=self.train_val_split_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,
        )
        self.sevir_val = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            num_shard=1,
            rank=0,
            start_date=self.train_val_split_date,
            end_date=self.train_test_split_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,
        )
        self.sevir_test = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            num_shard=1,
            rank=0,
            start_date=self.train_test_split_date,
            end_date=self.end_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,
        )
        self.sevir_predict = SEVIRTorchDataset(
            sevir_catalog=self.catalog_path,
            sevir_data_dir=self.raw_data_dir,
            raw_seq_len=self.raw_seq_len,
            split_mode="uneven",
            shuffle=False,
            seq_len=self.seq_len,
            stride=self.stride,
            sample_mode=self.sample_mode,
            batch_size=self.batch_size,
            layout=self.layout,
            num_shard=1,
            rank=0,
            start_date=self.train_test_split_date,
            end_date=self.end_date,
            output_type=self.output_type,
            preprocess=self.preprocess,
            rescale_method=self.rescale_method,
            verbose=self.verbose,
        )
        # print(">>> total samples =", len(self.all_samples))
        print(">>> train samples =", len(self.sevir_train))
        print(">>> val samples   =", len(self.sevir_val))
        print(">>> test samples  =", len(self.sevir_test))

    def train_dataloader(self):
        return self.sevir_train.get_torch_dataloader(
            outer_batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self.sevir_val.get_torch_dataloader(num_workers=self.num_workers)

    def test_dataloader(self):
        return self.sevir_test.get_torch_dataloader(num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.sevir_predict.get_torch_dataloader(num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)
