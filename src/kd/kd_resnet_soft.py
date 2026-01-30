# src/train_distill.py
import os, hydra, torch, logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = str(7)
os.environ["TORCH_HOME"] = "/home/sota/research/sotaohnuma/.cache/torch"
logging.basicConfig(filename="train.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Utils ======
def load_local_weights(model: nn.Module, path: str, strict: bool = False):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"weight not found: {path}")
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("model."):   nk = nk[len("model."):]
        if nk.startswith("module."):  nk = nk[len("module."):]
        new_sd[nk] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    if missing:
        logger.info(f"[load] missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        logger.info(f"[load] unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")

def kd_loss(logits_s: torch.Tensor, logits_t: torch.Tensor, T: float) -> torch.Tensor:
    """Temperature-scaled KL(student || teacher) * T^2"""
    with torch.no_grad():
        p_t = F.softmax(logits_t / T, dim=1)
    log_p_s = F.log_softmax(logits_s / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

# ====== 1 epoch (soft-label KD) ======
def run_epoch_distill(student, teacher, loader, opt, cfg_kd):
    train_mode = opt is not None
    (student.train() if train_mode else student.eval())
    teacher.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                logits_t = teacher(x)

            if train_mode:
                opt.zero_grad(set_to_none=True)
            logits_s = student(x)

            ce = F.cross_entropy(logits_s, y)                              # Hard-label
            kd = kd_loss(logits_s, logits_t, cfg_kd.T) if cfg_kd.alpha > 0 else 0.0
            loss = (1.0 - cfg_kd.alpha) * ce + cfg_kd.alpha * kd           # αでブレンド

            if train_mode:
                loss.backward()
                opt.step()

            loss_sum += loss.item() * y.size(0)
            pred = logits_s.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total

# ====== Main ======
@hydra.main(version_base=None, config_path="../configs", config_name="config_kd_resnet_soft")
def main(cfg: DictConfig):
    print("ORIGINAL CWD :", get_original_cwd())
    print("RUN CWD      :", os.getcwd())
    logger.info("ORIGINAL CWD : %s", get_original_cwd())
    logger.info("RUN CWD      : %s", os.getcwd())

    # --- Data ---
    tf = transforms.Compose([
        transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dl = DataLoader(ImageFolder(cfg.data.train_dir, tf),
                          batch_size=cfg.model.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(ImageFolder(cfg.data.val_dir, tf),
                          batch_size=cfg.model.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    test_dl  = DataLoader(ImageFolder(cfg.data.test_dir, tf),
                          batch_size=cfg.model.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # --- Student (Hydra instantiate) ---
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    BATCH_SIZE = model_cfg.pop("batch_size")
    LR = model_cfg.pop("lr")
    WEIGHT_DECAY = model_cfg.pop("weight_decay")
    student = instantiate(model_cfg, num_classes=cfg.num_classes).to(device)

    # --- Teacher: torchvision resnet50 + local .pt ---
    from torchvision import models as tvm
    teacher = tvm.resnet50(weights=None, num_classes=cfg.teacher.num_classes).to(device)
    if cfg.teacher.weights_path:
        load_local_weights(teacher, cfg.teacher.weights_path, strict=False)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # --- Optimizer (no scheduler) ---
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # --- Train Loop ---
    best_acc = 0.0
    tr_losses, tr_accs, vl_losses, vl_accs = [], [], [], []
    for epoch in range(cfg.train.epochs):
        logger.info(f"Start epoch {epoch+1}")
        tr_loss, tr_acc = run_epoch_distill(student, teacher, train_dl, opt, cfg.kd)
        vl_loss, vl_acc = run_epoch_distill(student, teacher, val_dl,   None, cfg.kd)

        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        vl_losses.append(vl_loss); vl_accs.append(vl_acc)
        print(f"[{epoch+1:02d}/{cfg.train.epochs}] train {tr_acc:.3%}/{tr_loss:.4f} | val {vl_acc:.3%}/{vl_loss:.4f}")
        logger.info(f"[{epoch+1:02d}/{cfg.train.epochs}] "
                    f"train {tr_acc:.3%}/{tr_loss:.4f} | "
                    f"val {vl_acc:.3%}/{vl_loss:.4f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(student.state_dict(), "best.pt")

    # --- Curves ---
    epochs = range(1, cfg.train.epochs + 1)
    plt.figure(); plt.plot(epochs, tr_losses, label="train"); plt.plot(epochs, vl_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss"); plt.savefig("loss_curve.png", dpi=150)

    plt.figure(); plt.plot(epochs, tr_accs, label="train"); plt.plot(epochs, vl_accs, label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy"); plt.savefig("accuracy_curve.png", dpi=150)

    # --- Test ---
    from utils import evaluate_model, plot_roc_curve, plot_pr_curve
    student.load_state_dict(torch.load("best.pt", map_location="cpu"))
    student.to(device).eval()
    metrics, curves = evaluate_model(student, test_dl, device)
    print("\n[Test Evaluation]")
    logger.info("\n[Test Evaluation]")
    for k, v in metrics.items():
        print(f" {k.capitalize():9}: {v:.3f}")
        logger.info(f" {k.capitalize():9}: {v:.3f}")

    fpr, tpr, _ = curves["roc"];  plot_roc_curve(fpr, tpr, metrics["auc"])
    recall, precision, _ = curves["pr"]; plot_pr_curve(recall, precision)

if __name__ == "__main__":
    main()

