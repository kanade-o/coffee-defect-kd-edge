import os, sys, logging, torch, pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.tuner import Tuner
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

HOMED = "/home/sota/research/sotaohnuma/coffee/"

# -----------------------------
# Data
# -----------------------------
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, input_size, batch_size):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir   = val_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1141, 0.1062, 0.0865],
                                 std =[0.2767, 0.2581, 0.2109])
        ])
    def setup(self, stage=None):
        self.train_ds = ImageFolder(self.train_dir, transform=self.tf)
        self.val_ds   = ImageFolder(self.val_dir,   transform=self.tf)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)

# -----------------------------
# LightningModules
# -----------------------------
class LitBatchFinder(pl.LightningModule):
    """Student only (通常学習)で最大BS探索に使う軽量モジュール"""
    def __init__(self, model, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = lr
    def forward(self, x): return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return self.loss_fn(logits, y)
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

class LitKDModule(pl.LightningModule):
    """蒸留時の最大BS探索用：教師はno-gradで前向きのみ"""
    def __init__(self, student, teacher,
                 lr: float = 1e-3, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.lr = lr
        self.T = temperature
        self.alpha = alpha
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, x): return self.student(x)

    def kd_loss(self, s_logits, t_logits, y):
        # KLDiv between softmax distributions (teacher/student), plus CE to labels
        T = self.T
        kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction="batchmean"
        ) * (T * T)
        ce = self.ce(s_logits, y)
        return self.alpha * kd + (1 - self.alpha) * ce

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            t_logits = self.teacher(x)
        s_logits = self.student(x)
        loss = self.kd_loss(s_logits, t_logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.student.parameters(), lr=self.lr)

# -----------------------------
# Helpers
# -----------------------------
def _instantiate_from_yaml(yaml_path: str, **extra):
    cfg = OmegaConf.load(yaml_path)
    cfg.pop("name", None)
    # 追加パラメータ（num_classesなど）を優先
    for k, v in extra.items():
        cfg[k] = v
    return instantiate(cfg)

def _load_teacher_from_yaml(yaml_path: str, weights_path: str, **extra):
    teacher = _instantiate_from_yaml(yaml_path, **extra)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Teacher weights not found: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    # 分類ヘッドを除外したい場合はstrict=False推奨
    teacher.load_state_dict(state, strict=False)
    return teacher

def _scale_bs_with_tuner(lit_module, dm, init_bs: int, output_dir: str) -> int:
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, logger=False, enable_checkpointing=False,
        default_root_dir=output_dir
    )
    tuner = Tuner(trainer)
    try:
        recommended = tuner.scale_batch_size(
            lit_module, datamodule=dm,
            mode="binsearch", init_val=init_bs, max_trials=13
        )
        return int(recommended)
    except RuntimeError as e:
        # OOMなどでbinsearchが失敗するケースの保険：1まで落として返す
        logging.exception(f"scale_batch_size failed: {e}")
        return 1

def find_fair_batch_sizes(
    model_name: str,
    student_cfg_path: str,
    train_dir: str, val_dir: str,
    input_size: int, init_bs: int,
    num_classes: int,
    output_dir: str,
    # 蒸留用
    teacher_cfg_path: str = None,
    teacher_weights_path: str = None,
    kd_temperature: float = 4.0,
    kd_alpha: float = 0.5,
):
    # Student instance
    student = _instantiate_from_yaml(student_cfg_path, num_classes=num_classes)

    # --- Student-only 最大バッチサイズ ---
    dm_student = ImageDataModule(train_dir, val_dir, input_size, init_bs)
    lit_student = LitBatchFinder(model=student, lr=1e-3)
    bs_student_max = _scale_bs_with_tuner(lit_student, dm_student, init_bs,
                                          os.path.join(output_dir, "student_only"))

    # --- KD 最大バッチサイズ（教師forward含む） ---
    if teacher_cfg_path and teacher_weights_path:
        # 教師は別インスタンスで読み込み（メモリ節約のため評価時のみ使用）
        teacher = _load_teacher_from_yaml(teacher_cfg_path, teacher_weights_path,
                                          num_classes=num_classes)
        # KD用にstudentを新規に作る（先のstudentと共有すると状態が変わる恐れがあるため）
        student_kd = _instantiate_from_yaml(student_cfg_path, num_classes=num_classes)
        dm_kd = ImageDataModule(train_dir, val_dir, input_size, init_bs)
        lit_kd = LitKDModule(student=student_kd, teacher=teacher,
                             lr=1e-3, temperature=kd_temperature, alpha=kd_alpha)
        bs_kd_max = _scale_bs_with_tuner(lit_kd, dm_kd, init_bs,
                                         os.path.join(output_dir, "distillation"))
    else:
        bs_kd_max = None

    # 公平比較の採用バッチサイズ：蒸留/非蒸留の両方で確実に動く値
    if bs_kd_max is not None:
        chosen = min(bs_student_max, bs_kd_max)
    else:
        chosen = bs_student_max

    logging.info(f"{model_name}: student_max={bs_student_max}, "
                 f"kd_max={bs_kd_max}, chosen={chosen}")
    return bs_student_max, bs_kd_max, chosen

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print("Usage: python batch_finder.py <GPU_ID>")
        sys.exit(1)
    gpu_id = args[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    train_dir   = HOMED + "data/bean_224/train"
    val_dir     = HOMED + "data/bean_224/val"
    input_size  = 224
    init_bs     = 16
    num_classes = 2

    confd = HOMED + "configs/model/"

    # 例：生徒/教師の設定（必要に応じて増やしてOK）
    # teacher_weights_path はローカルの学習済み重みを指すように設定してください
    tasks = [
        dict(
            model_name="cnn_student",
            student_cfg_path=os.path.join(confd, "cnn_student.yaml"),
            teacher_cfg_path=os.path.join(confd, "swin_s.yaml"),   # 例
            teacher_weights_path=HOMED + "src/models/weights/swin_small_finetuned.pth",
            kd_temperature=4.0,
            kd_alpha=0.5,
        ),
        # 追加したければここに辞書をさらに並べる
    ]

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_out = os.path.join("outputs", "batch_finder", now)
    os.makedirs(base_out, exist_ok=True)

    log_file = os.path.join(base_out, "summary.log")
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(f"START batch_finder at {now}")
    logging.info(f"train_dir={train_dir}, val_dir={val_dir}, input_size={input_size}, init_bs={init_bs}")

    summary_lines = []
    for t in tasks:
        model_name = t["model_name"]
        out_dir = os.path.join(base_out, model_name)
        os.makedirs(out_dir, exist_ok=True)

        s_max, kd_max, chosen = find_fair_batch_sizes(
            model_name=model_name,
            student_cfg_path=t["student_cfg_path"],
            train_dir=train_dir, val_dir=val_dir,
            input_size=input_size, init_bs=init_bs,
            num_classes=num_classes,
            output_dir=out_dir,
            teacher_cfg_path=t.get("teacher_cfg_path"),
            teacher_weights_path=t.get("teacher_weights_path"),
            kd_temperature=t.get("kd_temperature", 4.0),
            kd_alpha=t.get("kd_alpha", 0.5)
        )
        summary_lines.append(f"{model_name}: student_max={s_max}, kd_max={kd_max}, chosen={chosen}")

    summary_path = os.path.join(base_out, "batch_size_summary.txt")
    with open(summary_path, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")

    logging.info("Finished all models. Summary written.")
    print(f"Done! See {summary_path}")
