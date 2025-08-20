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
logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger()

def run_epoch(model, loader, loss_fn, opt=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if is_train:
                loss.backward()
                opt.step()

            loss_sum += loss.item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("ORIGINAL CWD :", get_original_cwd())
    print("RUN CWD      :", os.getcwd())
    logger.info("ORIGINAL CWD : %s", get_original_cwd()) 
    logger.info("RUN CWD      : %s", os.getcwd())

    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg_dict.pop("name", None)

    BATCH_SIZE = model_cfg_dict["batch_size"]
    print(f"batch: {BATCH_SIZE}, \n {model_cfg_dict}")
    logger.info(f"batch: {BATCH_SIZE}, \n {model_cfg_dict}")
    model_cfg_dict.pop("batch_size", None)


    LR = model_cfg_dict["lr"]
    print(f"lr: {LR}, \n {model_cfg_dict}")
    logger.info(f"lr: {LR}, \n {model_cfg_dict}")
    model_cfg_dict.pop("lr", None)

    WEIGHT_DECAY = model_cfg_dict["weight_decay"]
    print(f"weight_decay: {WEIGHT_DECAY}, \n {model_cfg_dict}")
    logger.info(f"weight_decay: {WEIGHT_DECAY}, \n {model_cfg_dict}")
    model_cfg_dict.pop("weight_decay", None)

    model = instantiate(model_cfg_dict, num_classes=cfg.num_classes)
    model = model.to(device)

    # ---------- DataLoader ----------
    tf = transforms.Compose([
        transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    # ---------- Model ----------
    if not hasattr(cfg, "model"):
        raise ValueError("必ず model=<モデル名> を指定してください")


    # ---------- Optimizer / Loss ----------
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    # ---------- ログ用リスト ----------
    tr_losses, tr_accs, vl_losses, vl_accs = [], [], [], []
    best_acc = 0.0

    # ---------- Epoch loop ----------
    for epoch in range(cfg.train.epochs):
        print(f"Start: {epoch}")
        logger.info(f"Start: {epoch}")
        tr_loss, tr_acc = run_epoch(model, train_dl, loss_fn, opt)
        vl_loss, vl_acc = run_epoch(model, val_dl,   loss_fn)

        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        vl_losses.append(vl_loss); vl_accs.append(vl_acc)

        print(f"[{epoch+1:02d}/{cfg.train.epochs}] "
              f"train {tr_acc:.3%}/{tr_loss:.4f} | "
              f"val {vl_acc:.3%}/{vl_loss:.4f}")
        logger.info(f"[{epoch+1:02d}/{cfg.train.epochs}] "
                    f"train {tr_acc:.3%}/{tr_loss:.4f} | "
                    f"val {vl_acc:.3%}/{vl_loss:.4f}")


        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), "best.pt")

    # ---------- Plot curves ----------
    epochs = range(1, cfg.train.epochs + 1)

    plt.figure()
    plt.plot(epochs, tr_losses, label="train")
    plt.plot(epochs, vl_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.savefig("loss_curve.png", dpi=150)

    plt.figure()
    plt.plot(epochs, tr_accs, label="train")
    plt.plot(epochs, vl_accs, label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy")
    plt.savefig("accuracy_curve.png", dpi=150)

    # --- Test ----
    model.load_state_dict(torch.load("best.pt"))
    metrics, curves = evaluate_model(model, test_dl, device)
    print("\n[Test Evaluation]")
    logger.info("\n[Test Evaluation]")
    for k, v in metrics.items():
        print(f" {k.capitalize():9}: {v:.3f}")
        logger.info(f" {k.capitalize():9}: {v:.3f}")

    fpr, tpr, _ = curves["roc"]
    plot_roc_curve(fpr, tpr, metrics["auc"])

    recall, precision, _ = curves["pr"]
    plot_pr_curve(recall, precision)

if __name__ == "__main__":
    main()

