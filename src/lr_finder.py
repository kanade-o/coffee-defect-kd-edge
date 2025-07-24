# src/train.py
import os
import copy 
import hydra
import torch
import logging
import optuna
import torch.nn as nn
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate, get_original_cwd
from sklearn.metrics import f1_score
from transformers import get_cosine_schedule_with_warmup

os.environ["TORCH_HOME"] = "/home/sota/research/sotaohnuma/.cache/torch"

logging.basicConfig(
    filename="lr.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger()

def evaluate(model, val_dl, device):
    print("starting eval")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return f1_score(all_labels, all_preds, average="binary", pos_label=1)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    num_epochs = 20
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    print(cfg.gpu)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("ORIGINAL CWD :", get_original_cwd())
    print("RUN CWD      :", os.getcwd())
    logger.info("ORIGINAL CWD : %s", get_original_cwd()) 
    logger.info("RUN CWD      : %s", os.getcwd())

    # ---------- Model ----------
    if not hasattr(cfg, "model"):
        raise ValueError("必ず model=<モデル名> を指定してください")

    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg_dict.pop("name", None)

    BATCH_SIZE = model_cfg_dict["batch_size"]
    print(f"batch: {BATCH_SIZE}, \n {model_cfg_dict}")
    model_cfg_dict.pop("batch_size", None)

    # ---------- DataLoader ----------
    tf = transforms.Compose([
        transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1141, 0.1062, 0.0865], std=[0.2767, 0.2581, 0.2109])
    ])

    print("Start loading train data")
    logger.info("Start loading train data")

    train_dl = DataLoader(
        ImageFolder(cfg.data.train_dir, tf),
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print("Start loading val data")
    logger.info("Start loading val data")
    val_dl = DataLoader(
        ImageFolder(cfg.data.val_dir, tf),
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("Start loading test data")
    logger.info("Start loading test data")
    test_dl = DataLoader(
        ImageFolder(cfg.data.test_dir, tf),
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("Finished DataLoader setup")
    logger.info("Finished DataLoader setup")

    def objective(trial):
        # ---------- Hyper Parameter ----------
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

        # ---------- Model ----------
        model = instantiate(model_cfg_dict, num_classes=cfg.num_classes)
        model = model.to(device)

        # ---------- Optimizer / criterion ----------
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
        criterion = nn.CrossEntropyLoss()

        # ---------- Scheduler ----------
        total_steps = num_epochs * len(train_dl)
        warmup_steps = int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
        )

        global_step = 0
        print(f"starting train")
        for epoch in range(num_epochs):
            print(f"epoch: {epoch}")
            model.train()
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1

        print("finished train")
        
        f1 = evaluate(model, val_dl, device)
        print(f"finished eval")
        return f1
    
    # run oputuna
    print("starting optimize")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Finished")
    for key, value in study.best_params.items():
        print(f" {key}: {value}")

if __name__ == "__main__":
    main()

