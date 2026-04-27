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
# from transformers import get_cosine_schedule_with_warmup  # 使わない前提

os.environ["CUDA_VISIBLE_DEVICES"] = str(9)
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
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # timmやLightning保存のprefix対策
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_sd[nk] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
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
    # fm: (N, C, H, W) あるいは (N, C) を受け取り得るがATは2D前提
    if fm.dim() == 2:
        # (N, C) -> (N, C, 1, 1) とみなす
        fm = fm[:, :, None, None]
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
    # (N, L, C) -> (N, C, H, W)  （ViT救済・今回はResNetなので通常不要）
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

# ========== 1 epoch（蒸留対応; 複数層AT対応） ==========

def run_epoch_distill(student, teacher, hooks, loader, opt, cfg_kd):
    """
    hooks: List[ (s_hook, t_hook, weight) ]
    train (opt!=None) / eval (opt=None) を兼ねる
    """
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

            kd = torch.tensor(0., device=device)
            if cfg_kd.alpha > 0:
                kd = kd_loss(logits_s, logits_t, cfg_kd.T)

            at_total = torch.tensor(0., device=device)
            if cfg_kd.beta > 0 and len(hooks) > 0:
                for s_hook, t_hook, w in hooks:
                    f_s, f_t = s_hook.out, t_hook.out
                    if f_s is None or f_t is None:
                        continue
                    if f_t.dim() == 3:  # ViT救済（今回は不要想定）
                        f_t = spatialize_tokens(f_t, grid_hw=cfg_kd.get("teacher_grid_hw", None))
                    at_total = at_total + w * at_loss(f_s, f_t)

            loss = ce + cfg_kd.alpha * kd + cfg_kd.beta * at_total

            if train_mode:
                loss.backward()
                opt.step()
                # if scheduler is not None:
                #     scheduler.step()

            loss_sum += loss.item() * y.size(0)
            pred = logits_s.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total

# ========== メイン ==========

@hydra.main(version_base=None, config_path="../configs", config_name="config_kd_resnet")
def main(cfg: DictConfig):
    global scheduler

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
    # student（Hydra instantiate）
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    BATCH_SIZE = model_cfg.pop("batch_size")
    LR = model_cfg.pop("lr")
    WEIGHT_DECAY = model_cfg.pop("weight_decay")
    student = instantiate(model_cfg, num_classes=cfg.num_classes).to(device)

    # teacher: torchvision resnet50 を固定（要: ローカル .pt）
    from torchvision import models as tvm
    teacher = tvm.resnet50(weights=None, num_classes=cfg.teacher.num_classes).to(device)
    if cfg.teacher.weights_path:
        load_local_weights(teacher, cfg.teacher.weights_path, strict=False)
    # teacher を固定
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # ---------- Hooks（複数層） ----------
    # cfg.kd.pairs: [{teacher_feat: "layer3", student_feat: "block3", weight: 1.0}, ...]
    hooks = []
    for pair in cfg.kd.pairs:
        t_mod = find_module(teacher, pair.teacher_feat)
        s_mod = find_module(student, pair.student_feat)
        t_hook, s_hook = FeatureHook(t_mod), FeatureHook(s_mod)
        hooks.append((s_hook, t_hook, float(pair.get("weight", 1.0))))

    # ---------- Optimizer（Schedulerなし） ----------
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---------- Train Loop ----------
    best_acc = 0.0
    tr_losses, tr_accs, vl_losses, vl_accs = [], [], [], []
    for epoch in range(cfg.train.epochs):
        logger.info(f"Start epoch {epoch+1}")
        tr_loss, tr_acc = run_epoch_distill(student, teacher, hooks, train_dl, opt, cfg.kd)
        vl_loss, vl_acc = run_epoch_distill(student, teacher, hooks, val_dl,   None, cfg.kd)

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

    # 後片付け
    for s_hook, t_hook, _ in hooks:
        s_hook.close(); t_hook.close()

if __name__ == "__main__":
    main()

