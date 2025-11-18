# eval_on_rasp.py
import os
import time
import csv

import torch
import torch.nn as nn
import psutil
import timm

DEVICE = "cpu"
N_WARMUP = 20
N_RUNS = 100

# =========================
#  Student model definition
# =========================

class cnn_student100(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super(cnn_student100, self).__init__()
        
        # ブロック1: 224x224 -> 112x112 -> 56x56
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112
        )
        
        # ブロック2: 56x56 -> 28x28
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28
        )
        
        # ブロック3: 28x28 -> 14x14
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14
        )
        
        # 分類器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

# =========================
#  Teacher models
# =========================

def create_swin_small(num_classes=2):
    return timm.create_model(
        "swin_small_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
    )

def create_resnet50(num_classes=2):
    return timm.create_model(
        "resnet50",
        pretrained=False,
        num_classes=num_classes,
    )

# =========================
#  Paths & model mapping
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# key, 表示名, アーキ種別, ファイル名
MODELS = [
    ("swin_teacher",      "Swin-Small Teacher",         "swin",     "swin_small.pt"),
    ("resnet_teacher",    "ResNet-50 Teacher",          "resnet",   "resnet50.pt"),
    ("student_plain",     "cnn_student100 (Student)",   "student",  "student.pt"),
    ("student_kd_swin",   "Student KD from Swin",       "student",  "distilled_from_swin.pt"),
    ("student_kd_resnet", "Student KD from ResNet50",   "student",  "distilled_from_resnet.pt"),
]

def build_arch(arch_type: str) -> nn.Module:
    if arch_type == "student":
        return cnn_student100(num_classes=2)
    elif arch_type == "swin":
        return create_swin_small(num_classes=2)
    elif arch_type == "resnet":
        return create_resnet50(num_classes=2)
    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")

def load_model(arch_type: str, filename: str) -> nn.Module:
    ckpt_path = os.path.join(BASE_DIR, filename)

    # まずは「とりあえず TorchScript 形式かも？」と仮定して試す
    try:
        model = torch.jit.load(ckpt_path, map_location=DEVICE)
        model.eval()
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        return model
    except Exception:
        # 失敗したら state_dict 形式とみなしてアーキを構築
        model = build_arch(arch_type)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        return model

# =========================
#  Dummy input
# =========================

def make_dummy_input():
    # 1x3x224x224
    return torch.randn(1, 3, 224, 224, device=DEVICE)

# =========================
#  Benchmark core
# =========================

def benchmark_with_resource(model: nn.Module,
                            x: torch.Tensor,
                            n_warmup: int = N_WARMUP,
                            n_runs: int = N_RUNS):
    proc = psutil.Process(os.getpid())

    # ウォームアップ
    for _ in range(n_warmup):
        _ = model(x)

    # CPU% 初回呼び出しは捨てる
    proc.cpu_percent(interval=None)

    times = []
    cpu_samples = []
    max_rss = 0

    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(x)
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

        cpu = proc.cpu_percent(interval=None)   # この区間のCPU%
        mem = proc.memory_info().rss           # bytes
        cpu_samples.append(cpu)
        if mem > max_rss:
            max_rss = mem

    times_sorted = sorted(times)
    avg_latency = sum(times) / len(times)
    p50_latency = times_sorted[len(times_sorted) // 2]
    p95_latency = times_sorted[int(len(times_sorted) * 0.95) - 1]
    avg_cpu = sum(cpu_samples) / len(cpu_samples)

    return {
        "avg_latency": avg_latency,        # seconds
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "avg_cpu_percent": avg_cpu,
        "max_rss_bytes": max_rss,
    }

# =========================
#  Main
# =========================

def main():
    x = make_dummy_input()

    out_csv = os.path.join(BASE_DIR, "benchmark_rpi5_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "key",
            "name",
            "avg_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "fps",
            "avg_cpu_percent",
            "max_rss_MB",
        ])

        for key, name, arch_type, fname in MODELS:
            print(f"\n=== Benchmarking {name} ({fname}) ===")

            if not os.path.exists(os.path.join(BASE_DIR, fname)):
                print(f"  -> {fname} が見つからないのでスキップします")
                continue

            model = load_model(arch_type, fname)
            result = benchmark_with_resource(model, x)

            avg_ms = result["avg_latency"] * 1000.0
            p50_ms = result["p50_latency"] * 1000.0
            p95_ms = result["p95_latency"] * 1000.0
            fps = 1.0 / result["avg_latency"] if result["avg_latency"] > 0 else 0.0
            max_rss_mb = result["max_rss_bytes"] / 1024.0 / 1024.0

            print(f"Average latency : {avg_ms:.2f} ms")
            print(f"Median latency  : {p50_ms:.2f} ms")
            print(f"p95 latency     : {p95_ms:.2f} ms")
            print(f"Throughput (FPS): {fps:.2f} frames/s")
            print(f"Avg CPU usage   : {result['avg_cpu_percent']:.1f} %")
            print(f"Max RSS         : {max_rss_mb:.2f} MB")

            writer.writerow([
                key,
                name,
                avg_ms,
                p50_ms,
                p95_ms,
                fps,
                result["avg_cpu_percent"],
                max_rss_mb,
            ])

    print(f"\nSaved results to: {out_csv}")

if __name__ == "__main__":
    main()

