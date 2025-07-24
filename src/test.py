# src/train.py
import os, copy, hydra, torch, logging, matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from hydra.utils import instantiate, get_original_cwd
from utils import evaluate_model, plot_roc_curve, plot_pr_curve

os.environ["TORCH_HOME"] = "/home/sota/research/sotaohnuma/.cache/torch"

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    # ---------- Model ----------
    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    print(type(model_cfg_dict))
    print(model_cfg_dict)
    hoge = model_cfg_dict["batch_size"]
    print(hoge)
    model_cfg_dict.pop("name", None)
    print(model_cfg_dict)

if __name__ == "__main__":
    main()

