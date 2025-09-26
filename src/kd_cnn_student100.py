# src/train_distill.py
import os, hydra, torch, logging, math
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

os.environ["CUDA_VISIBLE_DEVICES"] = str(8)
os.environ["TORCH_HOME"] = "/home/sota/research/sotaohnuma/.cache/torch"
logging.basicConfig(filename="train.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()

scheduler = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== KD / AT ユーティリティ ==========

def load_local_weights(model: nn.Module, path: str, strict: bool = False):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"weight not found: {path}")
    sd = torch.load(path, map_location="cpu")
    # timm/torchvision どちらでも動くようキーのラップを軽く許容
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        logger.info(f"[load] missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        logger.info(f"[load] unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")

class FeatureHook:
    def __init__(self, module: nn.Module):
        self.out = None
        self.h = module.register_forward_hook(self._hook)
    def _hook(self, m, i, o):
        self.out = o
    def close(self):
        self.h.remove()

def find_module(model: nn.Module, name: str) -> nn.Module:
    m = model
    for p in name.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m

def attention_map(fm: torch.Tensor, eps=1e-8):
    # fm: (N, C, H, W)
    am = (fm ** 2).sum(dim=1, keepdim=True)
    norm = am.norm(p=2, dim=(2,3), keepdim=True)
    return am / (norm + eps)

def at_loss(feat_s, feat_t):
    # 解像度合わせ（教師基準）
    if feat_s.shape[-2:] != feat_t.shape[-2:]:
        feat_s = F.adaptive_avg_pool2d(feat_s, feat_t.shape[-2:])
    return F.mse_loss(attention_map(feat_s), attention_map(feat_t).detach())

def kd_loss(logits_s, logits_t, T: float):
    p_t = F.softmax(logits_t / T, dim=1).detach()
    log_p_s = F.log_softmax(logits_s / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

def spatialize_tokens(x: torch.Tensor, grid_hw=None):
    # (N, L, C) -> (N, C, H, W)
    if x.dim() == 4:
        return x
    N, L, C = x.shape
    if grid_hw is None:
        s = int(math.sqrt(L))
        if s * s != L:
            raise ValueError(f"Token length {L} is not square; set teacher_grid_hw")
        H = W = s
    else:
        H, W = grid_hw
        assert H * W == L
    return x.transpose(1, 2).contiguous().view(N, C, H, W)

# ========== 1 epoch（蒸留対応） ==========

def run_epoch_distill(student, teacher, t_hook, s_hook, loader, opt, cfg_kd):
    """train (opt!=None) / eval (opt=None) を兼ねる"""
    train_mode = opt is not None
    (student.train() if train_mode else student.eval())
    teacher.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 教師 forward（勾配なし・固定）
            with torch.no_grad():
                logits_t = teacher(x)

            # 生徒 forward
            if train_mode:
                opt.zero_grad(set_to_none=True)
            logits_s = student(x)

            # ---- 損失計算 ----
            ce = F.cross_entropy(logits_s, y)
            kd = kd_loss(logits_s, logits_t, cfg_kd.T) if cfg_kd.alpha > 0 else torch.tensor(0., device=device)

            f_s = s_hook.out
            f_t = t_hook.out
            # ViTがトークン列の場合の救済
            if f_t.dim() == 3:
                f_t = spatialize_tokens(f_t, grid_hw=cfg_kd.teacher_grid_hw)

            at = at_loss(f_s, f_t) if cfg_kd.beta > 0 else torch.tensor(0., device=device)
            loss = ce + cfg_kd.alpha * kd + cfg_kd.beta * at

            if train_mode:
                loss.backward()
                opt.step()
                if scheduler is not None:
                    scheduler.step()

            loss_sum += loss.item() * y.size(0)
            pred = logits_s.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total

# ========== メイン ==========

@hydra.main(version_base=None, config_path="../configs", config_name="config_kd_cnn_student100")
def main(cfg: DictConfig):
    global scheduler
    os.environ["CUDA_VISIBLE_DEVICES"] = str(6)

    print("ORIGINAL CWD :", get_original_cwd())
    print("RUN CWD      :", os.getcwd())
    logger.info("ORIGINAL CWD : %s", get_original_cwd())
    logger.info("RUN CWD      : %s", os.getcwd())

    # ---------- Data ----------
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

    # ---------- Models ----------
    # student は既存の instantiate を活用
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    BATCH_SIZE = model_cfg.pop("batch_size")
    LR = model_cfg.pop("lr")
    WEIGHT_DECAY = model_cfg.pop("weight_decay")
    student = instantiate(model_cfg, num_classes=cfg.num_classes).to(device)

    # teacher は timm / torchvision どちらでもOK（Hydraで name/type を指定）
    teacher_cfg = OmegaConf.to_container(cfg.teacher, resolve=True)
    teacher_name = teacher_cfg.pop("name")
    teacher_type = teacher_cfg.pop("type")  # "timm" or "torchvision"
    teacher_w    = teacher_cfg.pop("weights_path")
    num_classes_t = teacher_cfg.pop("num_classes", cfg.num_classes)

    if teacher_type == "timm":
        import timm
        teacher = timm.create_model(teacher_name, pretrained=False, num_classes=num_classes_t)
    else:
        from torchvision import models as tvm
        teacher = getattr(tvm, teacher_name)(weights=None, num_classes=num_classes_t)
    teacher.to(device)

    # ローカル重みの読み込み（student/teacher 両方）
    #if cfg.student_weights_path:
    #    load_local_weights(student, cfg.student_weights_path, strict=False)
    if teacher_w:
        load_local_weights(teacher, teacher_w, strict=False)

    # teacher は固定
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher.eval()

    # ---------- Hooks ----------
    t_mod = find_module(teacher, cfg.kd.teacher_feat)
    s_mod = find_module(student, cfg.kd.student_feat)
    t_hook, s_hook = FeatureHook(t_mod), FeatureHook(s_mod)

    # ---------- Optimizer / Loss / Scheduler ----------
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = cfg.train.epochs * len(train_dl)
    warmup = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)

    # ---------- Train Loop ----------
    best_acc = 0.0
    tr_losses, tr_accs, vl_losses, vl_accs = [], [], [], []
    for epoch in range(cfg.train.epochs):
        logger.info(f"Start epoch {epoch+1}")
        tr_loss, tr_acc = run_epoch_distill(student, teacher, t_hook, s_hook, train_dl, opt, cfg.kd)
        vl_loss, vl_acc = run_epoch_distill(student, teacher, t_hook, s_hook, val_dl,   None, cfg.kd)

        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        vl_losses.append(vl_loss); vl_accs.append(vl_acc)
        print(f"[{epoch+1:02d}/{cfg.train.epochs}] train {tr_acc:.3%}/{tr_loss:.4f} | val {vl_acc:.3%}/{vl_loss:.4f}")
        logger.info(f"[{epoch+1:02d}/{cfg.train.epochs}] "
                    f"train {tr_acc:.3%}/{tr_loss:.4f} | "
                    f"val {vl_acc:.3%}/{vl_loss:.4f}")


        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(student.state_dict(), "best.pt")

    # ---------- Curves ----------
    epochs = range(1, cfg.train.epochs + 1)
    plt.figure(); plt.plot(epochs, tr_losses, label="train"); plt.plot(epochs, vl_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss"); plt.savefig("loss_curve.png", dpi=150)

    plt.figure(); plt.plot(epochs, tr_accs, label="train"); plt.plot(epochs, vl_accs, label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy"); plt.savefig("accuracy_curve.png", dpi=150)

    # ---------- Test ----------
    from utils import evaluate_model, plot_roc_curve, plot_pr_curve
    student.load_state_dict(torch.load("best.pt"))
    metrics, curves = evaluate_model(student, test_dl, device)
    print("\n[Test Evaluation]")
    logger.info("\n[Test Evaluation]")
    for k, v in metrics.items():
        print(f" {k.capitalize():9}: {v:.3f}")
        logger.info(f" {k.capitalize():9}: {v:.3f}")

    fpr, tpr, _ = curves["roc"];  plot_roc_curve(fpr, tpr, metrics["auc"])
    recall, precision, _ = curves["pr"]; plot_pr_curve(recall, precision)

    # 後片付け
    t_hook.close(); s_hook.close()

if __name__ == "__main__":
    main()

