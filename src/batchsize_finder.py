import os, torch, sys, logging, pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.tuner import Tuner
from omegaconf import OmegaConf
from hydra.utils import instantiate, get_original_cwd
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

HOMED = "/home/sota/research/sotaohnuma/coffee/"

class LitBatchFinder(pl.LightningModule):
    def __init__(self, model, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return self.loss_fn(logits, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
            transforms.Normalize(
                mean=[0.1141, 0.1062, 0.0865],
                std =[0.2767, 0.2581, 0.2109]
            )
        ])

    def setup(self, stage=None):
        self.train_ds = ImageFolder(self.train_dir, transform=self.tf)
        self.val_ds   = ImageFolder(self.val_dir,   transform=self.tf)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

def find_batch_size_for_model(model_name: str, model_cfg_path: str,
                              train_dir: str, val_dir: str,
                              input_size: int, init_bs: int,
                              num_classes: int,
                              output_dir: str):
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cfg.pop("name", None)
    model = instantiate(model_cfg, num_classes=num_classes)

    # LightningModule / DataModule
    lit = LitBatchFinder(model=model, lr=1e-3)
    dm  = ImageDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        input_size=input_size,
        batch_size=init_bs
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=output_dir
    )
    tuner = Tuner(trainer)

    recommended_bs = tuner.scale_batch_size(
        lit,
        datamodule=dm,
        mode="binsearch", 
        init_val=init_bs,
        max_trials=13
    )

    logging.info(f"{model_name}: recommended batch size → {recommended_bs}")
    return recommended_bs


if __name__ == "__main__":
    args = sys.argv
    if not args[1]:
        exit()
    gpu_id = args[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    train_dir   = HOMED + "data/bean_224/train"
    val_dir     = HOMED + "data/bean_224/val"
    input_size  = 224
    init_bs     = 16
    num_classes = 2
        
    confd = HOMED + "configs/model/"

    model_cfgs = {
#        "resnet18": confd + "resnet18.yaml",
#        "resnet50": confd + "resnet50.yaml",
#        "deit_b":   confd + "deit_b.yaml",
#        "deit_bd":   confd + "deit_bd.yaml",
#        "deit_small":   confd + "deit_small_patch16_224.yaml",
#        "CNN224":   confd + "cnn224.yaml",
#        "efficientnet": confd + "efficientnetb0.yaml",
#        "vit": confd + "vit.yaml",
#        "vit_3rd": confd + "vit_3rd.yaml",
#        "vit_two": confd + "vit_two.yaml",
#        "vit_b16": confd + "vit_b16.yaml",
#        "swin_b": confd + "swin_b.yaml",
#        "swin_s": confd + "swin_s.yaml",
#        "swin_t": confd + "swin_t.yaml",
        "mobilenetv2": confd + "mobilenetv2.yaml",
        "tiny_vit": confd + "tiny_vit.yaml",
    }

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_out = os.path.join("outputs", "batch_finder", now)
    os.makedirs(base_out, exist_ok=True)

    log_file = os.path.join(base_out, "summary.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(f"START batch_finder at {now}")
    logging.info(f"train_dir={train_dir}, val_dir={val_dir}, input_size={input_size}, init_bs={init_bs}")

    summary = {}
    for model_name, cfg_path in model_cfgs.items():
        logging.info(f"--- Processing model: {model_name} ---")
        out_dir = os.path.join(base_out, model_name)
        os.makedirs(out_dir, exist_ok=True)

        bs = find_batch_size_for_model(
            model_name=model_name,
            model_cfg_path=cfg_path,
            train_dir=train_dir,
            val_dir=val_dir,
            input_size=input_size,
            init_bs=init_bs,
            num_classes=num_classes,
            output_dir=out_dir
        )
        summary[model_name] = bs

    summary_path = os.path.join(base_out, "batch_size_summary.txt")
    with open(summary_path, "w") as f:
        for m, bs in summary.items():
            f.write(f"{m}: {bs}\n")
    logging.info("Finished all models. Summary written.")
    print(f"Done! See {summary_path}")

