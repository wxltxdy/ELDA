# -*- coding: utf-8 -*-
"""
各种效用函数
注意:许多功能是从Spotlight (MIT)复制过来的
"""
import logging
import os.path
from collections import UserDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import neptune.new as neptune
import numpy as np
import pandas as pd
import pyro
import seaborn as sns
import torch
import yaml

from .datasets import Interactions

_logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def gpu(tensor, gpu=False):
    if gpu:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):
    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size", 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


# ToDo: Check if this needs to be in numpy or better in pytorch?
def shuffle(*arrays, **kwargs):
    rng = np.random.default_rng(kwargs.get("rng"))

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have " "the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    rng.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def assert_no_grad(variable):
    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def set_seed(seed, cuda=False):
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)


# ToDo:检查这是否需要在numpy或更好的pytorch?
def sample_items(num_items, shape, rng=None):
    """Randomly sample a number of items"""
    rng = np.random.default_rng(rng)
    items = rng.integers(0, num_items, shape, dtype=np.int64)
    return items

#得到当前用户对应的(整个训练集中的所有的项目)的集合
def process_ids(user_ids, item_ids, n_items, use_cuda, cartesian):
    # 传入的item_ids是空的，直接预测所有的值。
    if item_ids is None:#所有物品的 ID 存储在 item_ids 变量中
        item_ids = np.arange(n_items, dtype=np.int64)
    # else:
    #     print(f"我们真正预测的项目的值不为空，具体大小为{item_ids.shape}")
    if np.isscalar(user_ids):
        user_ids = np.array(user_ids, dtype=np.int64)

    user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
    item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

    if cartesian:
        item_ids, user_ids = (
            item_ids.repeat(user_ids.size(0), 1),
            user_ids.repeat(1, item_ids.size(0)).view(-1, 1),
        )
    else:
        user_ids = user_ids.expand(item_ids.size(0), 1)

    user_var = gpu(user_ids, use_cuda)
    item_var = gpu(item_ids, use_cuda)

    # 函数返回两个一维张量  代表所有用户的id和所有需要预测的项目的id。
    return user_var.squeeze(), item_var.squeeze()


def relpath_to_abspath(path: Path, anchor_path: Path):
    if path.is_absolute():
        return path
    return (anchor_path / path).resolve()

# 提供一些工具函数来解析、读取和操作该配置文件
class Config(UserDict):
    def __init__(self, path: Path, **kwargs):
        super().__init__()
        self.path = path

        with open(path, "r") as fh:
            self.yaml_content = fh.read()

        cfg = yaml.safe_load(self.yaml_content)
        timestamp = datetime.now()
        # 稍后存储id的配置文件名称
        cfg["main"].setdefault("name", os.path.splitext(path.name)[0])
        cfg["main"]["path"] = path.parent
        cfg["main"]["log_level"] = getattr(logging, cfg["main"]["log_level"])
        cfg["main"]["timestamp"] = timestamp
        cfg["main"]["timestamp_str"] = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        cfg["main"].update(kwargs)
        self._resolve_paths(cfg, path.parent)

        sec_cfg = cfg["neptune"]
        if sec_cfg["api_token"].upper() != "ANONYMOUS":
            # 读取令牌文件并将其替换为配置
            with open(Path(sec_cfg["api_token"]).expanduser()) as fh:
                sec_cfg["api_token"] = fh.readline().strip()

        self.data.update(cfg)  # 把cfg设置为自己的字典

    def _resolve_paths(self, cfg: Dict[str, Any], anchor_path: Path):
        """使用' anchor_path '解析所有的相对路径"""
        for k, v in cfg.items():
            if isinstance(v, dict):
                self._resolve_paths(v, anchor_path)
            elif k.endswith("_path"):
                cfg[k] = relpath_to_abspath(Path(v).expanduser(), anchor_path)


def log_summary(df: pd.DataFrame):
    run = neptune.get_last_run()
    for _, row in df.iterrows():
        metric = row.pop("metric")
        for name, value in row.items():
            run[f"summary/{metric}_{name}"].log(value)

# 记录数据集的相关信息并将其上传到 Neptune 项目中
def log_dataset(name, interactions: Interactions):
    # get_last_run() 方法获取当前实验的最后一次运行记录
    run = neptune.get_last_run()
    run[f"data/{name}/hash"] = interactions.hash()
    # 我们计算数据集中实际唯一的实体!
    for prop_name, prop_val in [
        ("n_users", len(np.unique(interactions.user_ids))),
        ("n_items", len(np.unique(interactions.item_ids))),
        ("n_interactions", len(interactions)),
    ]:
        run[f"data/{name}/{prop_name}"] = prop_val


def cmp_ranks(orig_scores, alt_scores, eps=1e-4):
    """Compare ranking of scores forgiving rounding errors"""
    orig_ranks = np.argsort(orig_scores)
    alt_ranks = np.argsort(alt_scores)

    for idx in np.where(orig_ranks != alt_ranks)[0]:
        twin1 = orig_ranks[idx]
        twin2 = alt_ranks[idx]
        orig_delta = abs(orig_scores[twin1] - orig_scores[twin2])
        alt_delta = abs(alt_scores[twin1] - alt_scores[twin2])

        # false if permutation is not due to similar scores (-> rounding errors)
        if orig_delta + alt_delta > eps:
            return False
    return True


def reparam_beta(mu: float, scale: float) -> Tuple[float, float]:
    """Reparameterize Beta dist with mean `mu` and `scale` for scaling the variance

    The variance is in the interval `(0, mu * (1 - mu) )`. `scale` is thus in
    `(0, 1)` and is multiplied to `mu * (1 - mu)`.
    """
    assert 0 < scale < 1, "scale must be in (0, 1)!"
    var = scale * mu * (1 - mu)
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta = alpha * (1 / mu - 1)
    return alpha, beta


def reparam_beta_inv(alpha: float, beta: float) -> Tuple[float, float]:
    assert alpha > 0 and beta > 0, "alpha and beta must be > 0!"
    mu = alpha / (alpha + beta)
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    scale = var / (mu * (1 - mu))
    return mu, scale


def split_along_dim_apply(x, func, dim):
    """In the spirit of Numpy's apply_along_axis but differently.

    Splits along a dimension and applies a function to the remaining dims.
    """
    vals = [func(x_i) for x_i in torch.unbind(x, dim=dim)]
    if isinstance(vals[0], torch.Tensor):
        return torch.stack(vals)
    else:
        return torch.Tensor(vals)


def plot_cat(dist):
    """Plot a categorical distribution nicely"""
    sns.barplot(x=np.arange(dist.shape[0]), y=dist.numpy())
